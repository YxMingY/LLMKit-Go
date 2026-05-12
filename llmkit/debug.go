// debug.go
package llmkit

import (
	"fmt"
	"strings"
)

var TraceDebugEnabled = false

func traceDebug(format string, args ...any) {
	if !TraceDebugEnabled {
		return
	}
	fmt.Printf("[trace] "+format+"\n", args...)
}

func preview(text string, max int) string {
	return strings.TrimSpace(text)
	// if len(text) <= max {
	// 	return text
	// }
	// return text[:max] + " …(truncated)"
}
