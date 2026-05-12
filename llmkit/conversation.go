package llmkit

import (
	"context"
	"fmt"

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
	currentMsg     *MessageBuilder               // 当前正在构建的消息缓冲区
	MaxHistory     int                           // 滑动窗口大小
	SystemPrompt   *openai.ChatCompletionMessage // 可选的 system prompt，始终位于请求最前
	RequestOptions *RequestOptions               // 当前会话默认请求参数，由 Conversation 持有
}

// --- 便捷的链式添加方法 (直接操作内部缓冲区) ---

// AddText 向当前消息缓冲区添加文本
func (conv *Conversation) AddText(text string) *Conversation {
	conv.currentMsg.AddText(text)
	return conv
}

// AddImageURL 向当前消息缓冲区添加图片 URL
func (conv *Conversation) AddImageURL(url string) *Conversation {
	conv.currentMsg.AddImageURL(url)
	return conv
}

// AddImageBase64 向当前消息缓冲区添加 Base64 图片。
// 默认会按 image/png 处理；如需其他类型，可额外传入 MIME 类型。
func (conv *Conversation) AddImageBase64(data string, mimeType ...string) *Conversation {
	conv.currentMsg.AddImageBase64(data, mimeType...)
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
	conv.currentMsg = NewMessageBuilder()
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
	if len(conv.currentMsg.parts) == 0 {
		return "", fmt.Errorf("current message is empty, please AddText or AddImage first")
	}

	userMsg := conv.currentMsg.Build()

	// 2. 立即重置缓冲区，准备下一次输入 (复用或新建均可，这里选择新建以彻底隔离状态)
	conv.currentMsg = NewMessageBuilder()

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
	if len(conv.currentMsg.parts) == 0 {
		return fmt.Errorf("current message is empty")
	}

	userMsg := conv.currentMsg.Build()
	conv.currentMsg = NewMessageBuilder() // 发送即清空

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
