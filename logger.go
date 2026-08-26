// logger.go — logging with the same format as the Python version:
// "YYYY-MM-DD HH:MM:SS,ms LEVEL name: message" (Python asctime uses a comma before ms).

package proxycache

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	logLevelDebug = 10
	logLevelInfo  = 20
	logLevelWarn  = 30
	logLevelError = 40
	logLevelCrit  = 50
)

var (
	logThreshold = parseLogLevel(os.Getenv("LOG_LEVEL"))
	logMu        sync.Mutex
)

func parseLogLevel(s string) int {
	switch strings.ToUpper(s) {
	case "DEBUG":
		return logLevelDebug
	case "INFO":
		return logLevelInfo
	case "WARNING", "WARN":
		return logLevelWarn
	case "ERROR":
		return logLevelError
	case "CRITICAL", "FATAL":
		return logLevelCrit
	default:
		return logLevelInfo
	}
}

func levelName(l int) string {
	switch l {
	case logLevelDebug:
		return "DEBUG"
	case logLevelInfo:
		return "INFO"
	case logLevelWarn:
		return "WARNING"
	case logLevelError:
		return "ERROR"
	default:
		return "CRITICAL"
	}
}

func logf(lvl int, name, format string, args ...any) {
	if lvl < logThreshold {
		return
	}
	logMu.Lock()
	defer logMu.Unlock()
	now := time.Now()
	ts := now.Format("2006-01-02 15:04:05") + "," + fmt.Sprintf("%03d", now.Nanosecond()/1e6)
	os.Stderr.WriteString(fmt.Sprintf("%s %s %s: %s\n", ts, levelName(lvl), name, fmt.Sprintf(format, args...)))
}

func logDebug(name, format string, args ...any) { logf(logLevelDebug, name, format, args...) }
func logInfo(name, format string, args ...any)  { logf(logLevelInfo, name, format, args...) }
func logWarn(name, format string, args ...any)  { logf(logLevelWarn, name, format, args...) }
func logError(name, format string, args ...any) { logf(logLevelError, name, format, args...) }
