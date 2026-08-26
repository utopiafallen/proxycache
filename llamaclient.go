// llamaclient.go — HTTP client for llama.cpp: /v1/chat/completions (stream and
// non-stream), /slots save/restore, slot discovery (incl. router-mode fallback),
// and model discovery.
//
// Timeouts: plain requests use REQUEST_TIMEOUT, /slots?action=restore uses
// SLOT_TIMEOUT, and streaming requests use the caller's context only (never a
// deadline — a slow prompt prefill must not be killed mid-stream).

package proxycache

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	// ErrTimeout marks a request that hit its configured deadline.
	ErrTimeout = errors.New("request timed out")
	// ErrConn marks network-level failures (refused, reset, DNS, ...).
	ErrConn = errors.New("connection error")
)

// HTTPStatusError is the Go equivalent of httpx.HTTPStatusError: a non-2xx
// response raised by raise_for_status(). Carries the status code and body.
type HTTPStatusError struct {
	StatusCode int
	Body       string
}

func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf("backend returned status %d: %s", e.StatusCode, truncateString(e.Body, 512))
}

type LlamaClient struct {
	Mu               sync.Mutex
	baseURL          string
	httpClient       *http.Client
	requestCount     int
	connectionErrors int
}

// NewLlamaClient creates a client for a backend base URL.
func NewLlamaClient(baseURL string) *LlamaClient {
	c := &LlamaClient{baseURL: strings.TrimRight(baseURL, "/")}
	c.createClient()
	logInfo("llama_client", "Initialized HTTP client for %s", baseURL)
	return c
}

func (c *LlamaClient) createClient() {
	// Matches httpx.Limits(max_keepalive_connections=20, max_connections=100,
	// keepalive_expiry=30).
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     30 * time.Second,
	}
	c.httpClient = &http.Client{Transport: transport}
}

// Recreate drops the old client and abandons its pooled connections (they are
// reclaimed by the GC). The old pool is deliberately not awaited/closed:
// closing it can hang on half-open connections.
func (c *LlamaClient) Recreate() {
	c.Mu.Lock()
	defer c.Mu.Unlock()
	c.createClient()
}

func (c *LlamaClient) maybeRecreate() {
	c.Mu.Lock()
	defer c.Mu.Unlock()
	c.requestCount++
	if c.requestCount >= ClientRecreateInterval {
		logInfo("llama_client", "Recreating HTTP client for %s after %d requests, %d connection errors (interval=%d)",
			c.baseURL, c.requestCount, c.connectionErrors, ClientRecreateInterval)
		c.requestCount = 0
		c.connectionErrors = 0
		c.createClient()
	}
}

func (c *LlamaClient) client() *http.Client {
	c.Mu.Lock()
	defer c.Mu.Unlock()
	return c.httpClient
}

// Close releases idle pooled connections.
func (c *LlamaClient) Close() {
	c.Mu.Lock()
	defer c.Mu.Unlock()
	c.httpClient.CloseIdleConnections()
}

func (c *LlamaClient) BaseURL() string { return c.baseURL }

// doRequest issues a request against the backend, JSON-encoding body when
// non-nil, and classifies transport errors into ErrTimeout/ErrConn.
func (c *LlamaClient) doRequest(ctx context.Context, method, path string, query url.Values, body any) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reqBody = bytes.NewReader(data)
	}
	u := c.baseURL + path
	if query != nil {
		u += "?" + query.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, method, u, reqBody)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client().Do(req)
	if err != nil {
		err = classifyHTTPErr(err)
		if errors.Is(err, ErrConn) {
			c.Mu.Lock()
			c.connectionErrors++
			c.Mu.Unlock()
		}
		return nil, err
	}
	return resp, nil
}

// classifyHTTPErr maps net/http errors onto the Python httpx taxonomy the rest
// of the code expects: deadline -> ErrTimeout, any other url.Error -> ErrConn.
func classifyHTTPErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return ErrTimeout
	}
	var ue *url.Error
	if errors.As(err, &ue) {
		return ErrConn
	}
	return err
}

// withSlotID mirrors _with_slot_id: duplicates the slot pin across the body
// root, the options dict, and the query params.
func withSlotID(body map[string]any, slotID int) (map[string]any, url.Values) {
	b2 := make(map[string]any, len(body)+3)
	for k, v := range body {
		b2[k] = v
	}
	b2["_slot_id"] = slotID
	b2["slot_id"] = slotID
	b2["id_slot"] = slotID
	opts := map[string]any{}
	if o, ok := body["options"].(map[string]any); ok {
		for k, v := range o {
			opts[k] = v
		}
	}
	opts["slot_id"] = slotID
	opts["id_slot"] = slotID
	b2["options"] = opts
	s := strconv.Itoa(slotID)
	return b2, url.Values{"slot_id": {s}, "id_slot": {s}}
}

// ApplyChatTemplate POSTs /apply-template and returns the templated prompt.
func (c *LlamaClient) ApplyChatTemplate(ctx context.Context, messages []map[string]any) (string, error) {
	c.maybeRecreate()
	ctx2, cancel := context.WithTimeout(ctx, seconds(RequestTimeout))
	defer cancel()
	resp, err := c.doRequest(ctx2, "POST", "/apply-template", nil, map[string]any{"messages": messages})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return "", &HTTPStatusError{StatusCode: resp.StatusCode, Body: string(raw)}
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", err
	}
	if p, ok := data["prompt"].(string); ok {
		return p, nil
	}
	return "", nil
}

// Tokenize POSTs /tokenize and returns the token IDs.
func (c *LlamaClient) Tokenize(ctx context.Context, text string, addSpecial bool) ([]int, error) {
	c.maybeRecreate()
	ctx2, cancel := context.WithTimeout(ctx, seconds(RequestTimeout))
	defer cancel()
	resp, err := c.doRequest(ctx2, "POST", "/tokenize", nil, map[string]any{"content": text, "add_special": addSpecial})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return nil, &HTTPStatusError{StatusCode: resp.StatusCode, Body: string(raw)}
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	tokens := []int{}
	if list, ok := data["tokens"].([]any); ok {
		for _, t := range list {
			if n, ok := argInt(t); ok {
				tokens = append(tokens, n)
			}
		}
	}
	return tokens, nil
}

// ChatCompletions performs a non-streaming chat completion.
//
// A non-2xx response returns *HTTPStatusError (like raise_for_status). A
// 2xx response whose body is not JSON, or whose JSON is unparseable, returns
// an error-object dict with a nil error — exactly what the Python client
// returns ({"object": "error", ...}).
func (c *LlamaClient) ChatCompletions(ctx context.Context, body map[string]any, slotID int) (map[string]any, error) {
	c.maybeRecreate()
	body2, q := withSlotID(body, slotID)
	nMsg := 0
	if m, ok := body2["messages"].([]any); ok {
		nMsg = len(m)
	}
	logInfo("llama_client", "Chat completions: stream=false, body_stream=%v, %d messages", body2["stream"], nMsg)

	ctx2, cancel := context.WithTimeout(ctx, seconds(RequestTimeout))
	defer cancel()
	resp, err := c.doRequest(ctx2, "POST", "/v1/chat/completions", q, body2)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return nil, &HTTPStatusError{StatusCode: resp.StatusCode, Body: string(raw)}
	}

	ctype := resp.Header.Get("Content-Type")
	raw, _ := io.ReadAll(resp.Body)
	if !strings.Contains(ctype, "application/json") {
		logError("llama_client", "Non-JSON response from provider: content_type=%s, body length=%d", ctype, len(raw))
		return map[string]any{
			"object":  "error",
			"message": "provider returned non-JSON",
			"raw":     truncateString(string(raw), 2048),
		}, nil
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		logError("llama_client", "Failed to parse JSON response: status=%d, body length=%d, error=%s",
			resp.StatusCode, len(raw), err)
		return map[string]any{
			"object":  "error",
			"message": "invalid json from provider",
			"raw":     truncateString(string(raw), 2048),
		}, nil
	}
	return out, nil
}

// ChatCompletionsStream starts a streaming chat completion. The caller owns
// resp.Body and must close it. No deadline is applied to the request — only
// the caller's context (client disconnect) can cancel it.
func (c *LlamaClient) ChatCompletionsStream(ctx context.Context, body map[string]any, slotID int) (*http.Response, error) {
	c.maybeRecreate()
	body2, q := withSlotID(body, slotID)
	nMsg := 0
	if m, ok := body2["messages"].([]any); ok {
		nMsg = len(m)
	}
	logInfo("llama_client", "Chat completions: stream=true, body_stream=%v, %d messages", body2["stream"], nMsg)
	resp, err := c.doRequest(ctx, "POST", "/v1/chat/completions", q, body2)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// slotPath builds the /slots URL, using the llama-swap upstream form when
// BACKEND_MODE is "llama-swap" and a model name is given.
func (c *LlamaClient) slotPath(slotID int, modelName string) string {
	if BackendMode == "llama-swap" && modelName != "" {
		return "/upstream/" + url.PathEscape(modelName) + fmt.Sprintf("/slots/%d", slotID)
	}
	return fmt.Sprintf("/slots/%d", slotID)
}

// slotBody builds the save/restore JSON body. The "model" key is always
// present (null when no model name), matching the Python client.
func slotBody(basename, modelName string) map[string]any {
	body := map[string]any{"filename": basename, "model": nil}
	if modelName != "" {
		body["model"] = modelName
	}
	return body
}

// SaveSlot saves a slot's KV cache to disk.
//
// Returns (ok, size, err):
//   - network error        -> (false, 0, ErrConn/ErrTimeout)
//   - HTTP 500             -> (false, 0, nil)
//   - other non-2xx        -> (false, 0, *HTTPStatusError)  (raise_for_status)
//   - 2xx                  -> (true, n_written, nil)
func (c *LlamaClient) SaveSlot(ctx context.Context, slotID int, basename, modelName string) (bool, int, error) {
	c.maybeRecreate()
	ctx2, cancel := context.WithTimeout(ctx, seconds(RequestTimeout))
	defer cancel()
	resp, err := c.doRequest(ctx2, "POST", c.slotPath(slotID, modelName),
		url.Values{"action": {"save"}}, slotBody(basename, modelName))
	if err != nil {
		logWarn("llama_client", "Save slot failed: slot=%d, file=%s, error=%s", slotID, truncateString(basename, 16), err)
		return false, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 500 {
		logWarn("llama_client", "Save slot returned 500: slot=%d, file=%s", slotID, truncateString(basename, 16))
		return false, 0, nil
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return false, 0, &HTTPStatusError{StatusCode: resp.StatusCode, Body: string(raw)}
	}
	nWritten := 0
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err == nil {
		if n, ok := argInt(data["n_written"]); ok {
			nWritten = n
		}
	}
	return true, nWritten, nil
}

// RestoreSlot restores a KV cache into a slot, bounded by SLOT_TIMEOUT.
// Every failure (timeout, connection, non-200) returns false.
func (c *LlamaClient) RestoreSlot(ctx context.Context, slotID int, basename, modelName string) bool {
	c.maybeRecreate()
	ctx2, cancel := context.WithTimeout(ctx, seconds(SlotTimeout))
	defer cancel()
	resp, err := c.doRequest(ctx2, "POST", c.slotPath(slotID, modelName),
		url.Values{"action": {"restore"}}, slotBody(basename, modelName))
	if err != nil {
		if errors.Is(err, ErrTimeout) {
			logWarn("llama_client", "Restore slot timed out after %ds: slot=%d, file=%s",
				int(SlotTimeout), slotID, truncateString(basename, 16))
		}
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != 200 {
		logWarn("llama_client", "Restore slot failed: status=%d, slot=%d, file=%s",
			resp.StatusCode, slotID, truncateString(basename, 16))
		return false
	}
	return true
}

// GetSlotStatus GETs /slots/{slot_id}; returns the slot dict or nil on any error.
func (c *LlamaClient) GetSlotStatus(ctx context.Context, slotID int) map[string]any {
	c.maybeRecreate()
	ctx2, cancel := context.WithTimeout(ctx, seconds(RequestTimeout))
	defer cancel()
	resp, err := c.doRequest(ctx2, "GET", fmt.Sprintf("/slots/%d", slotID), nil, nil)
	if err != nil {
		logWarn("llama_client", "Failed to get slot status for slot %d: %s", slotID, err)
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		logWarn("llama_client", "Failed to get slot status for slot %d: status %d", slotID, resp.StatusCode)
		return nil
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil
	}
	return data
}

// GetSlotsInfo GETs /slots. On HTTP 400 (router mode) it falls back to
// querying each loaded child process's own /slots endpoint.
//
// Returns (slots, nil) on success, (nil, nil) on connection error (mirroring
// the Python warning path), and (nil, err) on other HTTP errors.
func (c *LlamaClient) GetSlotsInfo(ctx context.Context, modelName string) ([]map[string]any, error) {
	c.maybeRecreate()
	ctx2, cancel := context.WithTimeout(ctx, seconds(RequestTimeout))
	defer cancel()
	resp, err := c.doRequest(ctx2, "GET", "/slots", nil, nil)
	if err != nil {
		logWarn("llama_client", "Failed to get slots info from %s: %s", c.baseURL, err)
		return nil, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode == 400 {
		logInfo("llama_client", "Got 400 from /slots on %s — falling back to /models for router mode", c.baseURL)
		return c.getSlotsViaRouterModels(ctx2, modelName)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return nil, &HTTPStatusError{StatusCode: resp.StatusCode, Body: string(raw)}
	}
	var slots []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&slots); err != nil {
		return nil, err
	}
	return slots, nil
}

type routerModel struct {
	name string
	port int
}

// parseRouterModels GETs /models and returns loaded child-process models with
// their ports (from --port in status.args).
func (c *LlamaClient) parseRouterModels(ctx context.Context) []routerModel {
	c.maybeRecreate()
	resp, err := c.doRequest(ctx, "GET", "/models", nil, nil)
	if err != nil {
		logWarn("llama_client", "Failed to parse router models from %s: %s", c.baseURL, err)
		return nil
	}
	defer resp.Body.Close()
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil
	}
	list, _ := data["data"].([]any)
	var result []routerModel
	for _, item := range list {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		status, _ := m["status"].(map[string]any)
		if status == nil || status["value"] != "loaded" {
			continue
		}
		args, _ := status["args"].([]any)
		port := -1
		for i := 0; i+1 < len(args); i++ {
			if a, ok := args[i].(string); ok && a == "--port" {
				if p, ok := argInt(args[i+1]); ok {
					port = p
				}
				break
			}
		}
		if port < 0 {
			id, _ := m["id"].(string)
			logWarn("llama_client", "Router model '%s' has no port in status args", id)
			continue
		}
		name, _ := m["id"].(string)
		result = append(result, routerModel{name: name, port: port})
	}
	return result
}

// getChildSlots GETs /slots on a child process (always 127.0.0.1); nil on error.
func (c *LlamaClient) getChildSlots(ctx context.Context, childURL string) []map[string]any {
	c.maybeRecreate()
	req, err := http.NewRequestWithContext(ctx, "GET", childURL+"/slots", nil)
	if err != nil {
		return nil
	}
	resp, err := c.client().Do(req)
	if err != nil {
		logWarn("llama_client", "Failed to get slots from child process %s: %s", childURL, err)
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		logWarn("llama_client", "Child process %s returned status %d", childURL, resp.StatusCode)
		return nil
	}
	var slots []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&slots); err != nil {
		return nil
	}
	return slots
}

// getSlotsViaRouterModels queries slots from loaded child processes via GET /models.
// If modelName is given, only that model's child is queried.
func (c *LlamaClient) getSlotsViaRouterModels(ctx context.Context, modelName string) ([]map[string]any, error) {
	models := c.parseRouterModels(ctx)
	if len(models) == 0 {
		logWarn("llama_client", "No loaded models returned from /models on %s", c.baseURL)
		return nil, nil
	}
	if modelName != "" {
		var matched []routerModel
		var names []string
		for _, m := range models {
			names = append(names, m.name)
			if m.name == modelName {
				matched = append(matched, m)
			}
		}
		if len(matched) == 0 {
			logWarn("llama_client", "Model '%s' not loaded, available models: %s", modelName, strings.Join(names, ", "))
			return nil, nil
		}
		models = matched
	}
	var allSlots []map[string]any
	for _, m := range models {
		childURL := fmt.Sprintf("http://127.0.0.1:%d", m.port)
		slots := c.getChildSlots(ctx, childURL)
		if len(slots) > 0 {
			for _, s := range slots {
				s["_router_model"] = m.name
				s["_router_port"] = m.port
			}
			allSlots = append(allSlots, slots...)
			logWarn("llama_client", "Child process %s:%d returned %d slots", m.name, m.port, len(slots))
		} else {
			logWarn("llama_client", "Child process %s:%d returned no slots", m.name, m.port)
		}
	}
	if len(allSlots) > 0 {
		return allSlots, nil
	}
	logWarn("llama_client", "No slots found from any child process on %s", c.baseURL)
	return nil, nil
}

// DiscoveredModelInfo is a (name, n_ctx) pair from model discovery.
type DiscoveredModelInfo struct {
	Name string
	NCtx int
}

// DiscoverModels returns the models served by this backend as (name, n_ctx)
// pairs. Tries router mode (GET /models) first, then non-router
// (GET /v1/models, first entry only).
func (c *LlamaClient) DiscoverModels(ctx context.Context) ([]DiscoveredModelInfo, error) {
	c.maybeRecreate()
	ctx2, cancel := context.WithTimeout(ctx, seconds(RequestTimeout))
	defer cancel()

	// Router mode: GET /models
	resp, err := c.doRequest(ctx2, "GET", "/models", nil, nil)
	if err == nil {
		defer resp.Body.Close()
		var data map[string]any
		if jsonErr := json.NewDecoder(resp.Body).Decode(&data); jsonErr != nil {
			// JSONDecodeError is not an httpx.HTTPError in Python — it
			// propagates to the caller rather than falling through.
			return nil, jsonErr
		}
		if list, ok := data["data"].([]any); ok {
			var models []DiscoveredModelInfo
			for _, item := range list {
				entry, ok := item.(map[string]any)
				if !ok {
					continue
				}
				status, _ := entry["status"].(map[string]any)
				if status == nil || status["value"] != "loaded" {
					continue
				}
				name, _ := entry["id"].(string)
				nCtx := DefaultNCtx
				args, _ := status["args"].([]any)
				for i := 0; i+1 < len(args); i++ {
					a, ok := args[i].(string)
					if !ok {
						continue
					}
					if a == "-ctx" || a == "-c" || a == "--ctx-size" {
						if v, ok := argInt(args[i+1]); ok {
							nCtx = v
							break
						}
					}
				}
				if nCtx == DefaultNCtx {
					loaded, _ := status["loaded_info"].(map[string]any)
					if v, ok := argInt(loaded["n_ctx"]); ok {
						nCtx = v
					} else if v, ok := argInt(loaded["n_ctx_train"]); ok {
						nCtx = v
					}
				}
				if name != "" {
					models = append(models, DiscoveredModelInfo{Name: name, NCtx: nCtx})
				}
			}
			if len(models) > 0 {
				return models, nil
			}
			logWarn("llama_client", "Router /models returned no loaded models on %s", c.baseURL)
		}
	} else {
		logWarn("llama_client", "Router /models failed on %s: %s", c.baseURL, err)
	}

	// Non-router mode: GET /v1/models
	resp2, err2 := c.doRequest(ctx2, "GET", "/v1/models", nil, nil)
	if err2 == nil {
		defer resp2.Body.Close()
		var data map[string]any
		if jsonErr := json.NewDecoder(resp2.Body).Decode(&data); jsonErr == nil {
			if list, ok := data["data"].([]any); ok {
				for _, item := range list {
					entry, ok := item.(map[string]any)
					if !ok {
						continue
					}
					name, _ := entry["id"].(string)
					nCtx := DefaultNCtx
					if meta, ok := entry["meta"].(map[string]any); ok {
						if v, ok2 := argInt(meta["n_ctx"]); ok2 {
							nCtx = v
						}
					}
					if name != "" {
						return []DiscoveredModelInfo{{Name: name, NCtx: nCtx}}, nil
					}
				}
			}
		}
	} else {
		logWarn("llama_client", "Non-router /v1/models failed on %s: %s", c.baseURL, err2)
	}

	logWarn("llama_client", "discover_models returned empty on %s (both router and non-router failed)", c.baseURL)
	return nil, nil
}

// HealthCheck does GET /health with a 2s timeout. Any HTTP response (even 5xx)
// counts as up; only transport/timeout errors return non-nil.
func (c *LlamaClient) HealthCheck(ctx context.Context) error {
	ctx2, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	resp, err := c.doRequest(ctx2, "GET", "/health", nil, nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func seconds(v float64) time.Duration {
	return time.Duration(v * float64(time.Second))
}

// argInt converts a JSON scalar to int (numbers come back as float64).
func argInt(v any) (int, bool) {
	switch x := v.(type) {
	case string:
		n, err := strconv.Atoi(x)
		return n, err == nil
	case float64:
		return int(x), true
	case int:
		return x, true
	}
	return 0, false
}

func truncateString(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
