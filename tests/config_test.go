package tests

import (
	"os"
	"proxycache"
	"strings"
	"testing"
)

func TestEnvInt(t *testing.T) {
	cases := []struct {
		env  string
		def  int
		want int
	}{
		{"42", 7, 42},
		{"", 7, 7},
		{"abc", 7, 7},
		{"-5", 7, -5},
		{"0", 7, 0},
	}
	for _, c := range cases {
		t.Setenv("TEST_INT", c.env)
		if got := proxycache.EnvInt("TEST_INT", c.def); got != c.want {
			t.Errorf("EnvInt with env=%q def=%d = %d, want %d", c.env, c.def, got, c.want)
		}
	}
}

func TestEnvFloat(t *testing.T) {
	cases := []struct {
		env  string
		def  float64
		want float64
	}{
		{"1.5", 2, 1.5},
		{"", 2, 2},
		{"x", 2, 2},
		{"3", 2, 3},
	}
	for _, c := range cases {
		t.Setenv("TEST_FLOAT", c.env)
		if got := proxycache.EnvFloat("TEST_FLOAT", c.def); got != c.want {
			t.Errorf("EnvFloat with env=%q def=%v = %v, want %v", c.env, c.def, got, c.want)
		}
	}
}

func TestEnvStr(t *testing.T) {
	cases := []struct {
		env  string
		def  string
		want string
	}{
		{"val", "def", "val"},
		{"", "def", "def"},
	}
	for _, c := range cases {
		t.Setenv("TEST_STR", c.env)
		if got := proxycache.EnvStr("TEST_STR", c.def); got != c.want {
			t.Errorf("EnvStr with env=%q def=%q = %q, want %q", c.env, c.def, got, c.want)
		}
	}
}

func TestEnvBool(t *testing.T) {
	cases := []struct {
		env  string
		def  bool
		want bool
	}{
		{"", true, true},
		{"", false, false},
		{"true", false, true},
		{"TRUE", false, true},
		{"1", false, true},
		{"yes", false, true},
		{"false", true, false},
		{"0", true, false},
		{"no", true, false},
		{"whatever", true, false},
	}
	for _, c := range cases {
		t.Setenv("TEST_BOOL", c.env)
		if got := proxycache.EnvBool("TEST_BOOL", c.def); got != c.want {
			t.Errorf("EnvBool with env=%q def=%v = %v, want %v", c.env, c.def, got, c.want)
		}
	}
}

func TestSlotTimeoutDefault(t *testing.T) {
	if os.Getenv("SLOT_TIMEOUT") != "" {
		t.Skip("SLOT_TIMEOUT set in environment")
	}
	if proxycache.SlotTimeout != 30.0 {
		t.Errorf("SlotTimeout default = %v, want 30", proxycache.SlotTimeout)
	}
}

func TestShouldSaveCache(t *testing.T) {
	old := proxycache.CacheSaveRatioThreshold
	proxycache.CacheSaveRatioThreshold = 0.8
	defer func() { proxycache.CacheSaveRatioThreshold = old }()

	cases := []struct {
		ratio     float64
		recompute bool
		want      bool
	}{
		{0.9, false, false},
		{0.8, false, true},
		{0.79, false, true},
		{0.0, false, true},
		{0.9, true, true},
		{1.0, true, true},
	}
	for _, c := range cases {
		if got := proxycache.ShouldSaveCache(c.ratio, c.recompute); got != c.want {
			t.Errorf("ShouldSaveCache(ratio=%v, recompute=%v) = %v, want %v", c.ratio, c.recompute, got, c.want)
		}
	}
}

func TestShouldSkipSaveHeuristic(t *testing.T) {
	old := proxycache.CacheSaveCtxThreshold
	proxycache.CacheSaveCtxThreshold = 0.7
	defer func() { proxycache.CacheSaveCtxThreshold = old }()

	summarize := []map[string]any{{"role": "user", "content": "Please summarize this document"}}
	hello := []map[string]any{{"role": "user", "content": "Hello there"}}

	cases := []struct {
		tokens int
		nCtx   int
		msgs   []map[string]any
		want   bool
	}{
		{700, 1000, nil, true},
		{699, 1000, nil, false},
		{100, 0, nil, false},
		{10, 1000, summarize, true},
		{10, 1000, hello, false},
	}
	for _, c := range cases {
		got := proxycache.ShouldSkipSaveHeuristic(c.tokens, c.nCtx, c.msgs, nil)
		if got != c.want {
			t.Errorf("ShouldSkipSaveHeuristic(tokens=%d, nCtx=%d, msgs=%v) = %v, want %v",
				c.tokens, c.nCtx, c.msgs, got, c.want)
		}
	}
}

func TestClassifyRequest(t *testing.T) {
	cases := []struct {
		name  string
		msgs  []map[string]any
		want  float64
		signs []string
	}{
		{"empty", nil, 0, nil},
		{"single user greeting", []map[string]any{{"role": "user", "content": "Hello there"}}, 0.1, []string{"single_user_msg"}},
		{"summarize", []map[string]any{{"role": "user", "content": "Please summarize this document"}}, 0.4, []string{"single_user_msg", "keyword:summarize"}},
		{"with system prompt", []map[string]any{
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Please summarize this document"},
		}, 0.3, nil},
		{"conversation", []map[string]any{
			{"role": "user", "content": "What is the capital of France?"},
			{"role": "assistant", "content": "Paris is the capital and largest city of France."},
		}, 0.0, nil},
		{"long prompt", []map[string]any{{"role": "user", "content": strings.Repeat("word ", 1800)}}, 0.2, nil},
		{"parts content", []map[string]any{{
			"role":    "user",
			"content": []any{map[string]any{"type": "text", "text": "Summarize this"}},
		}}, 0.4, nil},
	}
	for _, c := range cases {
		got := proxycache.ClassifyRequest(c.msgs, nil)
		if got.Score != c.want {
			t.Errorf("%s: ClassifyRequest score = %v, want %v (signals %v)", c.name, got.Score, c.want, got.Signals)
		}
		for _, s := range c.signs {
			found := false
			for _, g := range got.Signals {
				if g == s {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("%s: missing signal %q in %v", c.name, s, got.Signals)
			}
		}
	}
}

func TestGetMsgText(t *testing.T) {
	cases := []struct {
		msg  map[string]any
		want string
	}{
		{map[string]any{"content": "Hello World"}, "hello world"},
		{map[string]any{}, ""},
		{map[string]any{"content": 42}, ""},
		{map[string]any{"content": []any{
			map[string]any{"type": "image"},
			map[string]any{"type": "text", "text": "Hi there"},
		}}, "hi there"},
		{map[string]any{"content": []any{map[string]any{"type": "text"}}}, ""},
		{map[string]any{"content": []any{"not-a-map"}}, ""},
	}
	for _, c := range cases {
		if got := proxycache.GetMsgText(c.msg); got != c.want {
			t.Errorf("GetMsgText(%v) = %q, want %q", c.msg, got, c.want)
		}
	}
}

func TestStripDelimited(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"before <text>abc</text> after", "before   after"},
		{"no tags", "no tags"},
		{"<a>x</a> and <b>y</b>", "  and  "},
	}
	for _, c := range cases {
		if got := proxycache.StripDelimited(c.in); got != c.want {
			t.Errorf("StripDelimited(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestExtractInstruction(t *testing.T) {
	if got := proxycache.ExtractInstruction("intro:\nrest"); got != "intro" {
		t.Errorf("ExtractInstruction colon+newline = %q, want %q", got, "intro")
	}
	if got := proxycache.ExtractInstruction("short"); got != "short" {
		t.Errorf("ExtractInstruction short = %q, want %q", got, "short")
	}
	long := strings.Repeat("a", 150)
	if got := proxycache.ExtractInstruction(long); got != long[:100] {
		t.Errorf("ExtractInstruction long = len %d, want 100", len(got))
	}
}

func TestClamp01(t *testing.T) {
	cases := []struct {
		in   float64
		want float64
	}{
		{-1, 0},
		{0, 0},
		{0.5, 0.5},
		{1, 1},
		{2, 1},
	}
	for _, c := range cases {
		if got := proxycache.Clamp01(c.in); got != c.want {
			t.Errorf("Clamp01(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestRoundTo3(t *testing.T) {
	cases := []struct {
		in   float64
		want float64
	}{
		{0.12345, 0.123},
		{0.12351, 0.124},
		{0, 0},
		{1, 1},
	}
	for _, c := range cases {
		if got := proxycache.RoundTo3(c.in); got != c.want {
			t.Errorf("RoundTo3(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}
