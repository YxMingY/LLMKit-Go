package test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"llmkit/llmkit"

	openai "github.com/sashabaranov/go-openai"
)

type conversationExportSnapshot struct {
	History      []openai.ChatCompletionMessage `json:"history"`
	CurrentParts []openai.ChatMessagePart       `json:"current_parts,omitempty"`
	MaxHistory   int                            `json:"max_history"`
	SystemPrompt *openai.ChatCompletionMessage  `json:"system_prompt,omitempty"`
}

type tracedExportSnapshot struct {
	conversationExportSnapshot
	Trace string `json:"trace"`
}

func TestConversationJSONRoundTrip(t *testing.T) {
	original := &llmkit.Conversation{
		History: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "hello"},
			{Role: openai.ChatMessageRoleAssistant, Content: "world"},
		},
		CurrentMsg: llmkit.NewMessageBuilder().AddText("draft text").AddImageURL("https://example.com/image.png"),
		MaxHistory: 7,
		SystemPrompt: &openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: "system prompt",
		},
		RequestOptions: &llmkit.RequestOptions{
			Temperature: 0.25,
			MaxTokens:   256,
			TopP:        0.8,
		},
	}

	data, err := original.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON failed: %v", err)
	}
	if !json.Valid([]byte(data)) {
		t.Fatalf("ExportJSON returned invalid JSON: %s", data)
	}
	assertConversationJSON(t, data, "draft text", "https://example.com/image.png")

	restored := &llmkit.Conversation{}
	if err := restored.ImportJSON(data); err != nil {
		t.Fatalf("ImportJSON failed: %v", err)
	}

	restoredData, err := restored.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON after import failed: %v", err)
	}
	if !jsonEqual(t, data, restoredData) {
		t.Fatalf("round-trip JSON mismatch\noriginal: %s\nrestored: %s", data, restoredData)
	}
	assertConversationJSON(t, restoredData, "draft text", "https://example.com/image.png")

	if got, want := len(restored.History), len(original.History); got != want {
		t.Fatalf("history length mismatch: got %d want %d", got, want)
	}
	if got, want := restored.MaxHistory, original.MaxHistory; got != want {
		t.Fatalf("max history mismatch: got %d want %d", got, want)
	}
	if restored.SystemPrompt == nil || original.SystemPrompt == nil {
		t.Fatalf("system prompt unexpectedly nil")
	}
	if got, want := restored.SystemPrompt.Content, original.SystemPrompt.Content; got != want {
		t.Fatalf("system prompt mismatch: got %q want %q", got, want)
	}
	if restored.RequestOptions == nil || original.RequestOptions == nil {
		t.Fatalf("request options unexpectedly nil")
	}
	if got, want := restored.RequestOptions.Temperature, original.RequestOptions.Temperature; got != want {
		t.Fatalf("temperature mismatch: got %v want %v", got, want)
	}
	if got, want := restored.RequestOptions.MaxTokens, original.RequestOptions.MaxTokens; got != want {
		t.Fatalf("max tokens mismatch: got %d want %d", got, want)
	}
	if got, want := restored.RequestOptions.TopP, original.RequestOptions.TopP; got != want {
		t.Fatalf("top-p mismatch: got %v want %v", got, want)
	}
}

func TestTracedConversationJSONRoundTrip(t *testing.T) {
	base := &llmkit.Conversation{
		History: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "alpha"},
		},
		CurrentMsg: llmkit.NewMessageBuilder().AddText("pending"),
		MaxHistory: 3,
		SystemPrompt: &openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: "trace prompt placeholder",
		},
		RequestOptions: &llmkit.RequestOptions{
			Temperature: 0.5,
			MaxTokens:   128,
			TopP:        0.95,
		},
	}

	baseData, err := base.ExportJSON()
	if err != nil {
		t.Fatalf("base ExportJSON failed: %v", err)
	}
	tracedData := addTraceField(t, baseData, "[TRACE_STATE]\nTopic: Go\n[/TRACE_STATE]")

	original := llmkit.NewTracedConversation(&llmkit.Conversation{}, nil)
	if err := original.ImportJSON(tracedData); err != nil {
		t.Fatalf("original ImportJSON failed: %v", err)
	}

	data, err := original.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON failed: %v", err)
	}
	if !json.Valid([]byte(data)) {
		t.Fatalf("ExportJSON returned invalid JSON: %s", data)
	}
	assertConversationJSON(t, data, "pending", "")
	assertTraceJSON(t, data, "[TRACE_STATE]\nTopic: Go\n[/TRACE_STATE]")

	restored := &llmkit.TracedConversation{Conversation: &llmkit.Conversation{}}
	if err := restored.ImportJSON(data); err != nil {
		t.Fatalf("ImportJSON failed: %v", err)
	}

	restoredData, err := restored.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON after import failed: %v", err)
	}
	if !jsonEqual(t, data, restoredData) {
		t.Fatalf("round-trip JSON mismatch\noriginal: %s\nrestored: %s", data, restoredData)
	}

	if got, want := restored.GetTrace(), original.GetTrace(); got != want {
		t.Fatalf("trace mismatch: got %q want %q", got, want)
	}
	if restored.SystemPrompt == nil {
		t.Fatalf("system prompt unexpectedly nil")
	}
	if got, want := restored.SystemPrompt.Content, llmkit.RenderSystemPrompt(original.GetTrace()); got != want {
		t.Fatalf("system prompt mismatch after trace restore")
	}
	assertConversationJSON(t, restoredData, "pending", "")
	assertTraceJSON(t, restoredData, "[TRACE_STATE]\nTopic: Go\n[/TRACE_STATE]")
	if got, want := len(restored.History), len(original.History); got != want {
		t.Fatalf("history length mismatch: got %d want %d", got, want)
	}

	plain := &llmkit.Conversation{}
	if err := plain.ImportJSON(data); err != nil {
		t.Fatalf("plain conversation import failed: %v", err)
	}
	plainData, err := plain.ExportJSON()
	if err != nil {
		t.Fatalf("plain ExportJSON failed: %v", err)
	}
	assertConversationJSON(t, plainData, "pending", "")
	if trace := extractTrace(t, plainData); trace != "" {
		t.Fatalf("plain conversation unexpectedly preserved trace: %q", trace)
	}
	if got, want := len(plain.History), len(original.History); got != want {
		t.Fatalf("plain import history mismatch: got %d want %d", got, want)
	}
}

func TestConversationAddImageAutoDetect(t *testing.T) {
	t.Run("url", func(t *testing.T) {
		conv := &llmkit.Conversation{CurrentMsg: llmkit.NewMessageBuilder()}
		conv.AddImage("https://example.com/image.png")

		snapshot := extractConversationSnapshot(t, mustExportJSON(t, conv))
		if got, want := len(snapshot.CurrentParts), 1; got != want {
			t.Fatalf("current parts length mismatch: got %d want %d", got, want)
		}
		if got, want := snapshot.CurrentParts[0].ImageURL.URL, "https://example.com/image.png"; got != want {
			t.Fatalf("url mismatch: got %q want %q", got, want)
		}
	})

	t.Run("file path", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "sample.png")
		if err := os.WriteFile(path, []byte("local-image-bytes"), 0o600); err != nil {
			t.Fatalf("write temp file failed: %v", err)
		}

		conv := &llmkit.Conversation{CurrentMsg: llmkit.NewMessageBuilder()}
		conv.AddImage(path)

		snapshot := extractConversationSnapshot(t, mustExportJSON(t, conv))
		if got, want := len(snapshot.CurrentParts), 1; got != want {
			t.Fatalf("current parts length mismatch: got %d want %d", got, want)
		}
		if snapshot.CurrentParts[0].ImageURL == nil {
			t.Fatalf("expected image url payload")
		}
		if !strings.HasPrefix(snapshot.CurrentParts[0].ImageURL.URL, "data:image/png;base64,") {
			t.Fatalf("expected base64 data url, got %q", snapshot.CurrentParts[0].ImageURL.URL)
		}
	})
}

func assertConversationJSON(t *testing.T, data string, expectedText string, expectedImageURL string) {
	t.Helper()

	snapshot := extractConversationSnapshot(t, data)
	if len(snapshot.CurrentParts) == 0 {
		t.Fatalf("expected current parts in JSON")
	}
	if expectedText != "" && snapshot.CurrentParts[0].Text != expectedText {
		t.Fatalf("pending text mismatch: got %q want %q", snapshot.CurrentParts[0].Text, expectedText)
	}
	if expectedImageURL != "" {
		if len(snapshot.CurrentParts) < 2 || snapshot.CurrentParts[1].ImageURL == nil {
			t.Fatalf("expected image part in JSON")
		}
		if got, want := snapshot.CurrentParts[1].ImageURL.URL, expectedImageURL; got != want {
			t.Fatalf("image URL mismatch: got %q want %q", got, want)
		}
	}
}

func assertTraceJSON(t *testing.T, data string, expectedTrace string) {
	t.Helper()

	trace := extractTrace(t, data)
	if got, want := trace, expectedTrace; got != want {
		t.Fatalf("trace mismatch in JSON: got %q want %q", got, want)
	}
}

func extractConversationSnapshot(t *testing.T, data string) conversationExportSnapshot {
	t.Helper()

	var snapshot conversationExportSnapshot
	if err := json.Unmarshal([]byte(data), &snapshot); err != nil {
		t.Fatalf("failed to parse conversation JSON: %v", err)
	}
	return snapshot
}

func extractTrace(t *testing.T, data string) string {
	t.Helper()

	var snapshot map[string]any
	if err := json.Unmarshal([]byte(data), &snapshot); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	if traceValue, ok := snapshot["trace"]; ok {
		if traceString, ok := traceValue.(string); ok {
			return traceString
		}
	}
	return ""
}

func addTraceField(t *testing.T, data string, trace string) string {
	t.Helper()

	var snapshot map[string]any
	if err := json.Unmarshal([]byte(data), &snapshot); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}
	snapshot["trace"] = trace
	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("failed to encode JSON: %v", err)
	}
	return string(encoded)
}

func mustExportJSON(t *testing.T, conv *llmkit.Conversation) string {
	t.Helper()

	data, err := conv.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON failed: %v", err)
	}
	return data
}

func jsonEqual(t *testing.T, left, right string) bool {
	t.Helper()

	var leftValue any
	if err := json.Unmarshal([]byte(left), &leftValue); err != nil {
		t.Fatalf("failed to parse left JSON: %v", err)
	}

	var rightValue any
	if err := json.Unmarshal([]byte(right), &rightValue); err != nil {
		t.Fatalf("failed to parse right JSON: %v", err)
	}

	return reflect.DeepEqual(leftValue, rightValue)
}
