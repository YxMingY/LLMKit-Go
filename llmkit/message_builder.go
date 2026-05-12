package llmkit

import (
	"fmt"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// --- 消息构建器 (MessageBuilder) ---

// MessageBuilder 用于链式构建单条用户消息的内容
type MessageBuilder struct {
	parts []openai.ChatMessagePart
}

// NewMessageBuilder 创建一个新的消息构建器
func NewMessageBuilder() *MessageBuilder {
	return &MessageBuilder{
		parts: make([]openai.ChatMessagePart, 0),
	}
}

// AddText 添加文本内容
func (b *MessageBuilder) AddText(text string) *MessageBuilder {
	b.parts = append(b.parts, openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeText,
		Text: text,
	})
	return b
}

// AddImageURL 添加图片 URL
func (b *MessageBuilder) AddImageURL(url string) *MessageBuilder {
	b.parts = append(b.parts, openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeImageURL,
		ImageURL: &openai.ChatMessageImageURL{
			URL:    url,
			Detail: openai.ImageURLDetailAuto,
		},
	})
	return b
}

// AddImageBase64 添加 Base64 图片。
// 默认使用 image/png；如需其他类型，可传入可选的 mimeType 覆盖。
func (b *MessageBuilder) AddImageBase64(data string, mimeType ...string) *MessageBuilder {
	mime := "image/png"
	if len(mimeType) > 0 && mimeType[0] != "" {
		mime = mimeType[0]
	}

	if strings.HasPrefix(data, "data:") {
		b.parts = append(b.parts, openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeImageURL,
			ImageURL: &openai.ChatMessageImageURL{
				URL:    data,
				Detail: openai.ImageURLDetailAuto,
			},
		})
		return b
	}

	base64Str := fmt.Sprintf("data:%s;base64,%s", mime, data)
	b.parts = append(b.parts, openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeImageURL,
		ImageURL: &openai.ChatMessageImageURL{
			URL:    base64Str,
			Detail: openai.ImageURLDetailAuto,
		},
	})
	return b
}

// Build 构建最终的消息对象
func (b *MessageBuilder) Build() openai.ChatCompletionMessage {
	// 优化：如果只有纯文本，使用 Content 字段以提高兼容性
	if len(b.parts) == 1 && b.parts[0].Type == openai.ChatMessagePartTypeText {
		return openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: b.parts[0].Text,
		}
	}

	// 多模态或混合内容使用 MultiContent
	return openai.ChatCompletionMessage{
		Role:         openai.ChatMessageRoleUser,
		MultiContent: b.parts,
	}
}
