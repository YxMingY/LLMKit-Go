package llmkit

import (
	"fmt"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// ProviderType 定义支持的厂商
type ProviderType string

const (
	ProviderOpenAI   ProviderType = "openai"
	ProviderZhipu    ProviderType = "zhipu"
	ProviderAliyun   ProviderType = "aliyun"
	ProviderMoonshot ProviderType = "moonshot"
	ProviderDeepSeek ProviderType = "deepseek"
)

// Config 客户端配置
type Config struct {
	Provider ProviderType
	APIKey   string
	Model    string
	BaseURL  string // 可选，为空则自动根据 Provider 设置
}

// Client 封装后的客户端
type Client struct {
	client *openai.Client
	model  string
}

// NewClient 创建一个新的 LLM 客户端
func NewClient(cfg Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("API Key cannot be empty")
	}

	config := openai.DefaultConfig(cfg.APIKey)

	// 自动设置 BaseURL
	if cfg.BaseURL == "" {
		switch strings.ToLower(string(cfg.Provider)) {
		case string(ProviderOpenAI):
			config.BaseURL = "https://api.openai.com/v1/"
		case string(ProviderZhipu):
			config.BaseURL = "https://open.bigmodel.cn/api/paas/v4/"
		case string(ProviderAliyun):
			config.BaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
		case string(ProviderMoonshot):
			config.BaseURL = "https://api.moonshot.cn/v1/"
		case string(ProviderDeepSeek):
			config.BaseURL = "https://api.deepseek.com/v1/"
		default:
			return nil, fmt.Errorf("unsupported provider: %s", cfg.Provider)
		}
	} else {
		config.BaseURL = cfg.BaseURL
	}

	return &Client{
		client: openai.NewClientWithConfig(config),
		model:  cfg.Model,
	}, nil
}

// NewConversation 创建会话
func (c *Client) NewConversation() *Conversation {
	return &Conversation{
		client:         c,
		History:        make([]openai.ChatCompletionMessage, 0),
		CurrentMsg:     NewMessageBuilder(), // 初始化一个空的构建器
		MaxHistory:     0,
		RequestOptions: DefaultOptions(),
	}
}

// NewTracedConversation 创建带思路摘要能力的会话。
// 由 Client 负责初始化底层 Conversation，可以避免调用方先手动 new Conversation。
func (c *Client) NewTracedConversation(policy TraceUpdatePolicy) *TracedConversation {
	return NewTracedConversation(c.NewConversation(), policy)
}
