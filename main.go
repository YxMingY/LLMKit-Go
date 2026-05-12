package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"llmkit/llmkit"
)

func main() {
	ctx := context.Background()

	llmkit.TraceDebugEnabled = true
	client, err := llmkit.NewClient(llmkit.Config{
		Provider: llmkit.ProviderAliyun,
		APIKey:   os.Getenv("QWEN_API_KEY"),
		Model:    "qwen3-omni-flash",
	})
	if err != nil {
		log.Fatal(err)
	}

	conv := client.NewTracedConversation(nil)
	conv.SetMaxHistory(6)

	fmt.Println("GLM 多模态交互程序")
	fmt.Println("输入 /help 查看命令，/quit 退出")

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				fmt.Println()
				return
			}
			log.Fatal(err)
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		cmd, arg := splitCommand(line)
		switch cmd {
		case "/help", "help", "?", "/h":
			printHelp()
		case "/quit", "quit", "/exit", "exit":
			return
		case "/clear":
			conv.ClearCurrentInput()
			fmt.Println("current input cleared")
		case "/history":
			fmt.Printf("history=%d max=%d\n", len(conv.History), conv.MaxHistory)
		case "/text", "text", "/t", "t":
			if arg == "" {
				fmt.Println("usage: /text 这里输入要发送的文字")
				continue
			}
			conv.AddText(arg)
			fmt.Println("text queued")
		case "/img", "img", "/image", "image", "/i", "i":
			if arg == "" {
				fmt.Println("usage: /img <image-url|local-file-path>")
				continue
			}
			if err := addImage(conv, arg); err != nil {
				fmt.Println("add image failed:", err)
				continue
			}
			fmt.Println("image queued")
		case "/send", "send", "/s", "s":
			resp, err := conv.Chat(ctx)
			if err != nil {
				fmt.Println("send failed:", err)
				continue
			}
			fmt.Println("AI:", resp)
		case "/stream", "stream", "/st", "st":
			fmt.Print("AI: ")
			err := conv.SendStream(ctx, func(chunk string) error {
				fmt.Print(chunk)
				return nil
			})
			fmt.Println()
			if err != nil {
				fmt.Println("stream failed:", err)
			}
		default:
			conv.AddText(line)
			fmt.Println("text queued")
		}
	}
}

func splitCommand(line string) (string, string) {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return "", ""
	}
	cmd := strings.ToLower(parts[0])
	if len(parts) == 1 {
		return cmd, ""
	}
	return cmd, strings.TrimSpace(line[len(parts[0]):])
}

func printHelp() {
	fmt.Println("命令:")
	fmt.Println("  /text <内容>       追加一段文本到当前待发送消息")
	fmt.Println("  /img <url|path>    追加一张图片。URL 直接发送，本地路径会转成 Base64")
	fmt.Println("  /send              发送当前积累的内容并获取普通回复")
	fmt.Println("  /stream            发送当前积累的内容并流式回显")
	fmt.Println("  /clear             清空当前待发送输入")
	fmt.Println("  /history           查看当前历史条数")
	fmt.Println("  /quit              退出程序")
	fmt.Println("  直接输入文本        等同于 /text <内容>")
}

func addImage(conv *llmkit.TracedConversation, source string) error {
	if strings.HasPrefix(source, "http://") || strings.HasPrefix(source, "https://") {
		conv.AddImageURL(source)
		return nil
	}

	data, err := os.ReadFile(filepath.Clean(source))
	if err != nil {
		return err
	}

	conv.AddImageBase64(base64.StdEncoding.EncodeToString(data))
	return nil
}
