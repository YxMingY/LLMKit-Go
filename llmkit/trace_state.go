// trace_state.go
package llmkit

const EmptyTraceState = `
[TRACE_STATE]
Topic:
Path:
- 
Stage:
Progress:
- 
Open Questions:
Constraints:
[/TRACE_STATE]
`
const TraceUpdaterSystemPrompt = `
你是一个内部【思维状态维护器】。

你的任务不是回答用户，而是维护 TRACE_STATE 的一致性和连续性。
你不会输出对话内容。
`

func RenderSystemPrompt(traceState string) string {
	return `
你是一个学习辅助 AI，与用户进行正常对话。

下面的 TRACE_STATE 是【系统内部维护的学习导航状态】：
- 用于帮助你判断当前学习阶段与讲解方向
- 不是用户输入的一部分
- 不是对话内容

严格规则：
- 回答时【绝不能】复述、解释或提及 TRACE_STATE
- 不要以任何形式输出其中的字段或内容
- 仅将其作为“隐含背景”来决定如何回答用户
- 用户只应看到自然、连贯的正常回答

[TRACE_STATE]
` + traceState
}

func BuildTraceUpdatePrompt(oldState string, recent string) string {
	return `
你将更新一个【思维状态 TRACE_STATE】。

核心原则：
- TRACE_STATE 是“渐进式更新”，不是重新生成
- 已出现过的学习阶段【不能删除】
- 新阶段只能在“明确完成上一个阶段后”追加到 Path

阶段判断规则：
- 如果 recent 只是围绕当前 Stage 进一步解释 → 不新增 Path
- 只有当 recent 明确开始讲解“下一个结构模块”时，才：
  1. 将旧 Stage 追加进 Path（如未存在）
  2. 设置新的 Stage

严禁：
- 丢弃旧 Path
- 将 Stage 回退或跳跃
- 重写 Topic

输出要求：
- 只输出更新后的 [TRACE_STATE]
- 保持格式完全一致
- 只输出更新后的 [TRACE_STATE]
- 你不会输出任何非 TRACE_STATE 的文本。
- 保持 TRACE_STATE 总长度尽量短

【当前 TRACE_STATE】
` + oldState + `

【最近对话】
` + recent
}
