package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

const MaxTraceLen = 800
const CompressTargetLen = 500

// TraceUpdatePolicy 控制何时触发思路摘要更新。可插拔以支持不同策略。
type TraceUpdatePolicy interface {
	ShouldUpdate(conv *Conversation) bool
	// OnUpdate 可以用于策略内部更新计数器等状态（可选）
	OnUpdate()
}

// CountPolicy 是一个简单的示例策略：每隔 Interval 次调用 Chat() 触发一次摘要更新。
// 这个策略只是为了演示，可被替换为更复杂的策略（基于 token 估算、主题漂移检测等）。
type CountPolicy struct {
	Interval int
	counter  int
}

func NewCountPolicy(interval int) *CountPolicy {
	if interval <= 0 {
		interval = 3
	}
	return &CountPolicy{Interval: interval, counter: 0}
}

func (p *CountPolicy) ShouldUpdate(conv *Conversation) bool {
	p.counter++
	traceDebug(
		"policy check: counter=%d interval=%d",
		p.counter,
		p.Interval,
	)
	return p.counter >= p.Interval
}

func (p *CountPolicy) OnUpdate() {
	traceDebug("policy reset counter")
	p.counter = 0
}

// TracedConversation 是对 Conversation 的装饰器。
// 这里使用匿名组合而不是再手写一层转发，目的是让 Conversation 的大部分方法自然“继承”到包装器上，
// 只对 Chat / Send / SendStream 这些真正需要插入 trace 逻辑的入口做覆盖。
type TracedConversation struct {
	*Conversation
	trace  string
	policy TraceUpdatePolicy
}

type tracedConversationJSONSnapshot struct {
	conversationJSONSnapshot
	Trace *string `json:"trace,omitempty"`
}

// NewTracedConversationFromConversation 允许在已有 Conversation 上启用 trace 能力。
func NewTracedConversation(base *Conversation, policy TraceUpdatePolicy) *TracedConversation {
	if policy == nil {
		policy = NewCountPolicy(3)
	}

	tc := &TracedConversation{
		Conversation: base,
		trace:        EmptyTraceState,
		policy:       policy,
	}

	// 初始化 system prompt
	base.SetSystemPrompt(RenderSystemPrompt(tc.trace))
	return tc
}

func recentMessagesText(
	history []openai.ChatCompletionMessage,
	n int,
) string {
	if n <= 0 || len(history) == 0 {
		traceDebug("recentMessages: empty")
		return ""
	}

	if len(history) > n {
		history = history[len(history)-n:]
	}

	traceDebug("recentMessages: using last %d messages", len(history))

	out := ""
	for i, m := range history {
		traceDebug(
			"recent[%d] role=%s len=%d",
			i,
			m.Role,
			len(m.Content),
		)
		out += fmt.Sprintf(
			"[%s]\n%s\n\n",
			m.Role,
			m.Content,
		)
	}
	return out
}

// maybeUpdateTrace 决定是否触发内部摘要请求；若触发则将摘要写入 base 的 system prompt。
// 设计原则：不修改 base.History（除非后续策略显式要求），并且对调用者完全透明。
func (t *TracedConversation) maybeUpdateTrace(ctx context.Context) error {
	traceDebug("maybeUpdateTrace: enter")

	if t.policy == nil {
		traceDebug("no policy, skip")
		return nil
	}

	if !t.policy.ShouldUpdate(t.Conversation) {
		traceDebug("policy decided: skip update")
		return nil
	}

	traceDebug("policy decided: UPDATE trace")

	recent := recentMessagesText(t.Conversation.History, 6)

	traceDebug(
		"old trace len=%d preview=\n%s",
		len(t.trace),
		preview(t.trace, 300),
	)

	updatePrompt := BuildTraceUpdatePrompt(t.trace, recent)

	traceDebug(
		"update prompt len=%d preview=\n%s",
		len(updatePrompt),
		preview(updatePrompt, 500),
	)

	req := openai.ChatCompletionRequest{
		Model: t.Conversation.client.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: TraceUpdaterSystemPrompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: updatePrompt,
			},
		},
		MaxTokens:   300,
		Temperature: 0.2,
	}

	traceDebug("sending trace update request")

	resp, err := t.Conversation.client.client.CreateChatCompletion(ctx, req)
	if err != nil {
		traceDebug("trace update ERROR: %v", err)
		return err
	}
	if len(resp.Choices) == 0 {
		traceDebug("trace update ERROR: empty response")
		return fmt.Errorf("trace update: empty response")
	}

	newState := strings.TrimSpace(resp.Choices[0].Message.Content)

	if len(newState) > MaxTraceLen {
		traceDebug(
			"trace too long (%d), compressing",
			len(newState),
		)

		newState = compressTrace(ctx, t, newState)
	}
	traceDebug(
		"new trace len=%d preview=\n%s",
		len(newState),
		preview(newState, 300),
	)

	// 更新 trace state
	t.trace = newState
	t.Conversation.SetSystemPrompt(RenderSystemPrompt(newState))

	traceDebug("system prompt updated")

	t.policy.OnUpdate()
	traceDebug("maybeUpdateTrace: done")

	return nil
}

func compressTrace(
	ctx context.Context,
	t *TracedConversation,
	trace string,
) string {
	prompt := `
请将下面的 TRACE_STATE 压缩为更短版本：

规则：
- 保留 Topic / Path / Stage
- Progress 用一句话概括
- 删除细节描述
- 输出必须是合法 TRACE_STATE
- 总长度不超过 500 字符

TRACE_STATE:
` + trace

	req := openai.ChatCompletionRequest{
		Model: t.Conversation.client.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "你是一个状态压缩器。",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   200,
		Temperature: 0.1,
	}

	resp, err := t.Conversation.client.client.CreateChatCompletion(ctx, req)
	if err != nil || len(resp.Choices) == 0 {
		traceDebug("compress failed, fallback to truncation")
		return trace[:CompressTargetLen]
	}

	out := strings.TrimSpace(resp.Choices[0].Message.Content)
	traceDebug("compressed trace len=%d", len(out))
	return out
}

// Chat 会在调用前根据策略决定是否静默更新思路摘要，随后调用底层 Conversation 的 Chat。
func (t *TracedConversation) Chat(ctx context.Context) (string, error) {
	// 尝试更新 trace；若失败，不阻塞主对话（返回值以主对话为准）
	_ = t.maybeUpdateTrace(ctx)
	return t.Conversation.Chat(ctx)
}

func (t *TracedConversation) Send(ctx context.Context) (string, error) {
	_ = t.maybeUpdateTrace(ctx)
	return t.Conversation.Send(ctx)
}

func (t *TracedConversation) SendStream(ctx context.Context, callback StreamCallback) error {
	_ = t.maybeUpdateTrace(ctx)
	return t.Conversation.SendStream(ctx, callback)
}

// GetTrace 返回当前缓存的思路摘要（只读，便于测试或调试）。
func (t *TracedConversation) GetTrace() string {
	return t.trace
}

// ExportJSON 将带 trace 的会话状态导出为 JSON 字符串。
func (t *TracedConversation) ExportJSON() (string, error) {
	trace := t.trace
	snapshot := tracedConversationJSONSnapshot{
		conversationJSONSnapshot: t.conversationSnapshot(),
		Trace:                    &trace,
	}

	payload, err := json.Marshal(snapshot)
	if err != nil {
		return "", err
	}
	return string(payload), nil
}

// ImportJSON 从 JSON 字符串恢复带 trace 的会话状态。
// 如果输入里不包含 trace 字段，则会恢复为默认的 EmptyTraceState。
func (t *TracedConversation) ImportJSON(data string) error {
	var snapshot tracedConversationJSONSnapshot
	if err := json.Unmarshal([]byte(data), &snapshot); err != nil {
		return err
	}

	t.Conversation.applyConversationSnapshot(snapshot.conversationJSONSnapshot)
	if snapshot.Trace != nil {
		t.trace = *snapshot.Trace
	} else {
		t.trace = EmptyTraceState
	}
	t.Conversation.SetSystemPrompt(RenderSystemPrompt(t.trace))
	return nil
}
