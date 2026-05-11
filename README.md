# glm-demo

一个基于 Go 的多模态大模型调用示例项目，封装了 `llmkit` 客户端，支持文本、图片 URL、Base64 图片和流式输出。

## 特性

- 支持多个厂商的 OpenAI 兼容接口：OpenAI、智谱、阿里云、Moonshot、DeepSeek
- 支持图文混合输入
- 支持普通对话和流式输出
- 支持会话历史和滑动窗口，避免上下文无限增长

## 项目结构

- [main.go](main.go) - 演示如何创建会话、发送图片并进行多轮对话
- [llmkit/client.go](llmkit/client.go) - 客户端初始化与厂商配置
- [llmkit/conversation.go](llmkit/conversation.go) - 会话管理、历史维护、普通/流式发送
- [llmkit/messagebuilder.go](llmkit/messagebuilder.go) - 消息构建器，支持文本和图片拼接
- [test.png](test.png) - 示例图片资源

## 运行环境

- Go 1.26.1 或更高版本
- 一个可用的模型 API Key

## 环境变量

当前示例默认使用智谱 AI，因此需要先设置：

- `ZHIPU_API_KEY`

如果你切换到其他厂商，需要改动 `main.go` 中的 `Provider` 和对应的 Key，例如：

- 阿里云：`DASHSCOPE_API_KEY`
- OpenAI：`OPENAI_API_KEY`
- Moonshot：`MOONSHOT_API_KEY`
- DeepSeek：`DEEPSEEK_API_KEY`

## 运行方式

```bash
go run .
```

程序会：

1. 初始化一个 `llmkit` 客户端
2. 创建一个会话对象
3. 发送一张图片 URL 给模型
4. 继续追问并演示多轮上下文
5. 使用流式输出打印模型回复

## `llmkit` 用法示例

下面是一个更完整的示例，和当前 `main.go` 的流程基本一致：先发一张图片 URL，再继续追问，最后演示本地 `test.png` 转 Base64 后的流式输出。

```go
package main

import (
    "context"
    "encoding/base64"
    "fmt"
    "log"
    "os"

    "glm-demo/llmkit"
)

func main() {
    ctx := context.Background()

    client, err := llmkit.NewClient(llmkit.Config{
        Provider: llmkit.ProviderZhipu,
        APIKey:   os.Getenv("ZHIPU_API_KEY"),
        Model:    "glm-4v-flash",
    })
    if err != nil {
        log.Fatal(err)
    }

    convo := client.NewConversation()
    convo.SetMaxHistory(6)

    convo.
        AddText("这张图里有什么？").
        AddImageURL("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=1024")

    resp1, err := convo.Send(ctx)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("AI:", resp1)

    convo.AddText("它看起来开心吗？")
    resp2, err := convo.Send(ctx)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("AI:", resp2)

    imageData, err := os.ReadFile("test.png")
    if err != nil {
        log.Fatal(err)
    }

    base64Str := base64.StdEncoding.EncodeToString(imageData)
    convo.
        AddText("请描述这张本地图片").
        AddImageBase64(base64Str)

    fmt.Print("AI: ")
    err = convo.SendStream(ctx, func(chunk string) error {
        fmt.Print(chunk)
        return nil
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println()
}
```

如果你只想发 Base64 图片，可以直接这样写：

```go
imageData, _ := os.ReadFile("test.png")
base64Str := base64.StdEncoding.EncodeToString(imageData)

convo.AddText("描述这张图片").AddImageBase64(base64Str)

// 如果不是 PNG，也可以指定 MIME 类型：
// convo.AddImageBase64(base64Str, "image/jpeg")
```

## 设计说明

- `Conversation` 会把每次发送的用户消息和模型回复都加入历史
- `SetMaxHistory(n)` 会限制历史消息数量，避免上下文过长
- `SendStream` 会边接收边回调，适合直接输出到终端

## 注意事项

- 代码里使用的是 OpenAI 兼容接口，模型名和 BaseURL 需要和对应厂商匹配
- 图片 URL 需要模型服务端可访问；如果是本地图片，建议先转成 Base64 再发送
- 如果你修改了 `main.go` 里的 provider，请同步更新环境变量名称

## 许可证

未指定许可证。如需开源分发，请补充相应 LICENSE 文件。