package llmkit

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// --- 请求配置 (RequestOptions) ---

type RequestOptions struct {
	Temperature float32
	MaxTokens   int
	TopP        float32
}

func DefaultOptions() *RequestOptions {
	return &RequestOptions{
		Temperature: 0.7,
		MaxTokens:   1024,
		TopP:        1.0,
	}
}

func (r *RequestOptions) WithTemperature(t float32) *RequestOptions {
	r.Temperature = t
	return r
}

func (r *RequestOptions) WithMaxTokens(m int) *RequestOptions {
	r.MaxTokens = m
	return r
}

func (r *RequestOptions) WithTopP(p float32) *RequestOptions {
	r.TopP = p
	return r
}

// --- 会话管理器 (Conversation) ---

// Conversation 现在兼具“会话历史管理”和“当前消息构建”的功能
type Conversation struct {
	client         *Client
	History        []openai.ChatCompletionMessage
	CurrentMsg     *MessageBuilder               // 当前正在构建的消息缓冲区
	MaxHistory     int                           // 滑动窗口大小
	SystemPrompt   *openai.ChatCompletionMessage // 可选的 system prompt，始终位于请求最前
	RequestOptions *RequestOptions               // 当前会话默认请求参数，由 Conversation 持有
}

type conversationJSONSnapshot struct {
	History        []openai.ChatCompletionMessage `json:"history"`
	CurrentParts   []openai.ChatMessagePart       `json:"current_parts,omitempty"`
	MaxHistory     int                            `json:"max_history"`
	SystemPrompt   *openai.ChatCompletionMessage  `json:"system_prompt,omitempty"`
	RequestOptions *RequestOptions                `json:"request_options,omitempty"`
}

func cloneRequestOptions(opts *RequestOptions) *RequestOptions {
	if opts == nil {
		return nil
	}
	clone := *opts
	return &clone
}

func (conv *Conversation) conversationSnapshot() conversationJSONSnapshot {
	parts := []openai.ChatMessagePart{}
	if conv.CurrentMsg != nil {
		parts = append(parts, conv.CurrentMsg.parts...)
	}

	return conversationJSONSnapshot{
		History:        append([]openai.ChatCompletionMessage(nil), conv.History...),
		CurrentParts:   parts,
		MaxHistory:     conv.MaxHistory,
		SystemPrompt:   conv.SystemPrompt,
		RequestOptions: cloneRequestOptions(conv.RequestOptions),
	}
}

func (conv *Conversation) applyConversationSnapshot(snapshot conversationJSONSnapshot) {
	conv.History = append([]openai.ChatCompletionMessage(nil), snapshot.History...)
	conv.MaxHistory = snapshot.MaxHistory

	if snapshot.SystemPrompt != nil {
		prompt := *snapshot.SystemPrompt
		conv.SystemPrompt = &prompt
	} else {
		conv.SystemPrompt = nil
	}

	if snapshot.RequestOptions != nil {
		conv.RequestOptions = cloneRequestOptions(snapshot.RequestOptions)
	} else {
		conv.RequestOptions = nil
	}

	conv.CurrentMsg = &MessageBuilder{parts: append([]openai.ChatMessagePart(nil), snapshot.CurrentParts...)}
	if conv.CurrentMsg == nil {
		conv.CurrentMsg = NewMessageBuilder()
	}
}

// ExportJSON 将当前会话状态导出为 JSON 字符串，便于重启后恢复上下文。
func (conv *Conversation) ExportJSON() (string, error) {
	payload, err := json.Marshal(conv.conversationSnapshot())
	if err != nil {
		return "", err
	}
	return string(payload), nil
}

// ImportJSON 从 JSON 字符串恢复会话状态。
func (conv *Conversation) ImportJSON(data string) error {
	var snapshot conversationJSONSnapshot
	if err := json.Unmarshal([]byte(data), &snapshot); err != nil {
		return err
	}
	conv.applyConversationSnapshot(snapshot)
	return nil
}

// --- 便捷的链式添加方法 (直接操作内部缓冲区) ---

// AddText 向当前消息缓冲区添加文本
func (conv *Conversation) AddText(text string) *Conversation {
	conv.CurrentMsg.AddText(text)
	return conv
}

// AddImageURL 向当前消息缓冲区添加图片 URL
func (conv *Conversation) AddImageURL(url string) *Conversation {
	conv.CurrentMsg.AddImageURL(url)
	return conv
}

// AddImage 自动识别输入是图片 URL 还是本地文件路径。
// URL 直接发送；本地文件会读取后转为 Base64 再发送。
func (conv *Conversation) AddImage(source string) *Conversation {
	if strings.HasPrefix(source, "http://") || strings.HasPrefix(source, "https://") {
		return conv.AddImageURL(source)
	}

	if strings.HasPrefix(source, "data:") {
		return conv.AddImageBase64(source)
	}

	data, err := os.ReadFile(filepath.Clean(source))
	if err == nil {
		conv.AddImageBase64(base64.StdEncoding.EncodeToString(data))
		return conv
	}

	return conv.AddImageBase64(source)
}

// AddImageBase64 向当前消息缓冲区添加 Base64 图片。
// 默认会按 image/png 处理；如需其他类型，可额外传入 MIME 类型。
func (conv *Conversation) AddImageBase64(data string, mimeType ...string) *Conversation {
	conv.CurrentMsg.AddImageBase64(data, mimeType...)
	return conv
}

// SetMaxHistory 设置滑动窗口
func (conv *Conversation) SetMaxHistory(n int) *Conversation {
	conv.MaxHistory = n
	conv.trimHistory()
	return conv
}

// SetRequestOptions 设置整个请求参数对象；传 nil 时恢复默认值。
func (conv *Conversation) SetRequestOptions(opts *RequestOptions) *Conversation {
	if opts == nil {
		conv.RequestOptions = DefaultOptions()
		return conv
	}
	clone := *opts
	conv.RequestOptions = &clone
	return conv
}

// SetTemperature 设置会话级温度参数。
func (conv *Conversation) SetTemperature(t float32) *Conversation {
	conv.ensureRequestOptions()
	conv.RequestOptions.Temperature = t
	return conv
}

// SetMaxTokens 设置会话级最大输出 token。
func (conv *Conversation) SetMaxTokens(m int) *Conversation {
	conv.ensureRequestOptions()
	conv.RequestOptions.MaxTokens = m
	return conv
}

// SetTopP 设置会话级 top-p 采样参数。
func (conv *Conversation) SetTopP(p float32) *Conversation {
	conv.ensureRequestOptions()
	conv.RequestOptions.TopP = p
	return conv
}

// SetSystemPrompt 设置或替换 system prompt。system message 始终位于发送给模型的消息最前端，
// 不计入历史裁剪逻辑，也不会被当做 user/assistant 历史处理。
func (conv *Conversation) SetSystemPrompt(text string) {
	if text == "" {
		conv.SystemPrompt = nil
		return
	}
	conv.SystemPrompt = &openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: text,
	}
}

// ClearCurrentInput 手动清空当前输入缓冲区（通常不需要手动调用，Send 会自动处理）
func (conv *Conversation) ClearCurrentInput() {
	conv.CurrentMsg = NewMessageBuilder()
}

// trimHistory 内部裁剪逻辑
func (conv *Conversation) trimHistory() {
	if conv.MaxHistory > 0 && len(conv.History) > conv.MaxHistory {
		conv.History = conv.History[len(conv.History)-conv.MaxHistory:]
	}
}

func (conv *Conversation) ensureRequestOptions() {
	if conv.RequestOptions == nil {
		conv.RequestOptions = DefaultOptions()
	}
}

func (conv *Conversation) currentRequestOptions() *RequestOptions {
	conv.ensureRequestOptions()
	clone := *conv.RequestOptions
	return &clone
}

// historyWithSystem 返回用于发送给 LLM 的消息切片：若存在 SystemPrompt，则始终放在最前。
func (conv *Conversation) historyWithSystem() []openai.ChatCompletionMessage {
	if conv.SystemPrompt == nil {
		// 返回 History 的拷贝以避免外部修改
		msgs := make([]openai.ChatCompletionMessage, len(conv.History))
		copy(msgs, conv.History)
		return msgs
	}
	msgs := make([]openai.ChatCompletionMessage, 0, 1+len(conv.History))
	msgs = append(msgs, *conv.SystemPrompt)
	msgs = append(msgs, conv.History...)
	return msgs
}

// --- 发送方法 (自动读取缓冲区并清空) ---

// Send 发送当前缓冲区的内容，并自动清空缓冲区
func (conv *Conversation) Send(ctx context.Context) (string, error) {
	// 1. 从缓冲区构建消息
	if len(conv.CurrentMsg.parts) == 0 {
		return "", fmt.Errorf("current message is empty, please AddText or AddImage first")
	}

	userMsg := conv.CurrentMsg.Build()

	// 2. 立即重置缓冲区，准备下一次输入 (复用或新建均可，这里选择新建以彻底隔离状态)
	conv.CurrentMsg = NewMessageBuilder()

	// 3. 加入历史
	conv.History = append(conv.History, userMsg)
	conv.trimHistory()

	// 4. 配置与请求
	options := conv.currentRequestOptions()

	req := openai.ChatCompletionRequest{
		Model:       conv.client.model,
		Messages:    conv.historyWithSystem(),
		Temperature: options.Temperature,
		MaxTokens:   options.MaxTokens,
		TopP:        options.TopP,
	}

	resp, err := conv.client.client.CreateChatCompletion(ctx, req)
	if err != nil {
		// 出错回滚：把刚才加进去的用户消息删掉，因为没得到回复，逻辑上不算完成一轮
		conv.History = conv.History[:len(conv.History)-1]
		return "", fmt.Errorf("chat error: %w", err)
	}

	if len(resp.Choices) == 0 {
		conv.History = conv.History[:len(conv.History)-1]
		return "", fmt.Errorf("no response")
	}

	// 5. 助手回复加入历史
	assistantMsg := resp.Choices[0].Message
	assistantMsg.Role = openai.ChatMessageRoleAssistant
	conv.History = append(conv.History, assistantMsg)

	return assistantMsg.Content, nil
}

// StreamCallback 流式输出回调函数类型
type StreamCallback func(chunk string) error

// SendStream 流式发送，同样自动清空缓冲区
func (conv *Conversation) SendStream(ctx context.Context, callback StreamCallback) error {
	if len(conv.CurrentMsg.parts) == 0 {
		return fmt.Errorf("current message is empty")
	}

	userMsg := conv.CurrentMsg.Build()
	conv.CurrentMsg = NewMessageBuilder() // 发送即清空

	conv.History = append(conv.History, userMsg)
	conv.trimHistory()

	options := conv.currentRequestOptions()

	req := openai.ChatCompletionRequest{
		Model:       conv.client.model,
		Messages:    conv.historyWithSystem(),
		Temperature: options.Temperature,
		MaxTokens:   options.MaxTokens,
		TopP:        options.TopP,
		Stream:      true,
	}

	stream, err := conv.client.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		conv.History = conv.History[:len(conv.History)-1]
		return fmt.Errorf("stream error: %w", err)
	}
	defer stream.Close()

	var fullResponse string
	for {
		resp, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			conv.History = conv.History[:len(conv.History)-1]
			return err
		}
		if len(resp.Choices) > 0 {
			content := resp.Choices[0].Delta.Content
			if content != "" {
				fullResponse += content
				if err := callback(content); err != nil {
					return err
				}
			}
		}
	}

	conv.History = append(conv.History, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: fullResponse,
	})

	return nil
}

// Chat 是 Send 的别名，便于使用更语义化的方法名（兼容外部示例中使用 Chat 的习惯）
func (conv *Conversation) Chat(ctx context.Context) (string, error) {
	return conv.Send(ctx)
}
