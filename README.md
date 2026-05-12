# llmkit

一个基于 Go 的多模态大模型调用库，封装了 `llmkit` 客户端，支持文本、图片 URL、本地图片转 Base64 和流式输出。

## 特性

- 支持多个厂商的 OpenAI 兼容接口：OpenAI、智谱、阿里云、Moonshot、DeepSeek
- 支持文本和图片混合输入
- 支持普通对话、流式输出和多轮历史
- 支持会话滑动窗口，避免上下文无限增长
- 支持把本地图片路径直接转换为 Base64 后发送
- 支持可选的 traced conversation 包装，用于自动维护内部 TRACE_STATE

## 入口程序

当前 `main.go` 是一个交互式 CLI，不再是固定演示脚本。启动后可以通过命令积累待发送内容，再统一发给模型。

### 命令

- `/text <内容>`: 追加一段文本到当前消息
- `/img <url|path>`: 追加一张图片，URL 直接发送，本地路径会读取并转成 Base64
- `/send`: 发送当前积累的内容并获取普通回复
- `/stream`: 发送当前积累的内容并流式回显
- `/clear`: 清空当前待发送输入
- `/history`: 查看当前历史条数
- `/quit`: 退出程序

直接输入文本等同于 `/text <内容>`。

## 项目结构

- [main.go](main.go) - 交互式命令行入口，支持文本、图片和流式输出
- [llmkit/client.go](llmkit/client.go) - 客户端初始化与厂商配置
- [llmkit/conversation.go](llmkit/conversation.go) - 会话管理、历史维护、普通/流式发送
- [llmkit/message_builder.go](llmkit/message_builder.go) - 消息构建器，支持文本、图片 URL 和 Base64 图片
- [llmkit/traced_conversation.go](llmkit/traced_conversation.go) - 带 TRACE_STATE 自动更新的会话包装
- [llmkit/trace_state.go](llmkit/trace_state.go) - TRACE_STATE 模板和 system prompt 生成
- [test/test.go](test/test.go) - 额外的多模态示例程序

## 运行环境

- Go 1.26.1 或更高版本
- 一个可用的模型 API Key

## 环境变量

当前 `main.go` 默认使用阿里云百炼，因此需要先设置：

- `QWEN_API_KEY`

如果你切换到其他厂商，需要改动 `main.go` 中的 `Provider` 和对应的 Key，例如：

- 智谱：`ZHIPU_API_KEY`
- OpenAI：`OPENAI_API_KEY`
- Moonshot：`MOONSHOT_API_KEY`
- DeepSeek：`DEEPSEEK_API_KEY`

## 运行方式

```bash
go run .
```

程序启动后可以先输入文本和图片，再用 `/send` 或 `/stream` 发送。

## `llmkit` 用法示例

下面是一个和当前 API 对齐的最小示例，展示了文本、图片 URL、Base64 图片和流式输出。

```go
package main

import (
    "context"
    "encoding/base64"
    "fmt"
    "log"
    "os"

    "llmkit/llmkit"
)

func main() {
    ctx := context.Background()

    client, err := llmkit.NewClient(llmkit.Config{
        Provider: llmkit.ProviderAliyun,
        APIKey:   os.Getenv("QWEN_API_KEY"),
        Model:    "qwen3-omni-flash",
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

如果图片不是 PNG，也可以显式指定 MIME 类型：

```go
convo.AddImageBase64(base64Str, "image/jpeg")
```

**TracedConversation — 长期知识脉络与思维保持**

`TracedConversation` 不只是一次性的上下文注入，而是为多轮、多阶段的对话提供一种“长期知识脉络”和“思维保持”能力：

- **长期知识脉络（Knowledge Thread）**：在跨多轮或多会话的学习/讲解场景中，`TracedConversation` 可保持话题进展、已覆盖的子模块与未完成事项，从而避免重复讲解与断裂式回复。
- **思维与进度保持（Persistent Thought State）**：它能保存当前的教学阶段、进度和开放问题，使模型在后续提问时延续先前的讲解语境，支持渐进式教学与分步演示。
- **个性化与目标导向**：通过记录已达成的目标和用户尚未掌握的点，模型能更好地调整后续输出的深度、举例风格与教学节奏，形成“个性化学习路线”。
- **场景示例**：适合教学、循序渐进的教程、长期项目辅导、复杂决策讨论或任何需要跨会话维持主题连贯性的场景。
- **用户可见性**：TRACE_STATE 为内部导航信息，不直接展示给用户；用户看到的是更连贯、少重复且更符合先前进度的回答。

简单使用示例：

```go
conv := client.NewTracedConversation(nil)
conv.SetMaxHistory(6)
// 之后按常规使用 AddText/AddImage/Send/SendStream 即可
```

如果你想了解系统如何保证状态渐进与一致，可以参考实现文件：
- [llmkit/traced_conversation.go](llmkit/traced_conversation.go#L1)
- [llmkit/trace_state.go](llmkit/trace_state.go#L1-L200)

## 设计说明

- `Conversation` 会把每次发送的用户消息和模型回复都加入历史
- `SetMaxHistory(n)` 会限制历史消息数量，避免上下文过长
- `SendStream` 会边接收边回调，适合直接输出到终端
- `TracedConversation` 会在发送前静默更新内部状态，并把它注入 system prompt

## 注意事项

- 代码里使用的是 OpenAI 兼容接口，模型名和 BaseURL 需要和对应厂商匹配
- 图片 URL 需要模型服务端可访问；如果是本地图片，CLI 会自动读取并转成 Base64
- 如果你修改了 `main.go` 里的 provider，请同步更新环境变量名称

## 许可证

未指定许可证。如需开源分发，请补充相应 LICENSE 文件。