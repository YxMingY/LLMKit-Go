package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"glm-demo/llmkit" // 请替换为你的实际模块名
)

func main() {
	ctx := context.Background()

	// 1. 初始化客户端
	client, err := llmkit.NewClient(llmkit.Config{
		Provider: llmkit.ProviderZhipu,
		APIKey:   os.Getenv("ZHIPU_API_KEY"),
		Model:    "glm-4v-flash",
	})
	if err != nil {
		log.Fatal(err)
	}

	// 2. 创建会话
	convo := client.NewConversation()

	// 【关键】设置滑动窗口：只保留最近 6 条消息（即最近 3 轮对话）
	// 这样可以防止 Token 爆炸，同时保留短期记忆
	convo.SetMaxHistory(6)

	// --- 第 1 轮：看图 ---
	fmt.Println("=== Round 1: Identify Image ===")
	convo.
		AddText("这张图里有什么？").
		AddImageURL("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=1024")

	resp1, _ := convo.Send(ctx)
	fmt.Println("AI:", resp1)

	// --- 第 2 轮：追问 ---
	fmt.Println("\n=== Round 2: Follow-up Question ===")
	convo.
		AddText("它看起来开心吗？")

	resp2, _ := convo.Send(ctx)
	fmt.Println("AI:", resp2)

	// --- 第 3 轮：继续追问 ---
	fmt.Println("\n=== Round 3: Another Question ===")
	convo.
		AddText("给它起个名字吧。")

	// 使用流式输出
	fmt.Print("AI: ")
	convo.SendStream(ctx, func(chunk string) error {
		fmt.Print(chunk)
		return nil
	})
	fmt.Println()

	// --- 第 4 轮：测试记忆遗忘 ---
	// 由于设置了 MaxHistory=6，此时第 1 轮的消息可能已经被移出窗口
	// 取决于具体实现，如果第1轮被移出，模型可能不再记得图片细节
	fmt.Println("\n=== Round 4: Test Memory (Old Context might be lost) ===")
	convo.
		AddText("我第一轮问的是什么问题？")

	resp4, _ := convo.Send(ctx)
	fmt.Println("AI:", resp4)

	fmt.Printf("\n当前历史消息条数: %d (限制为 6)\n", len(convo.History))
}
