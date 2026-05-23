import json
import os
from typing import TypedDict, Annotated, List, Optional
import operator

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS

load_dotenv()

# ---- State ----
class ResearchState(TypedDict):
    question: str
    queries: Annotated[List[str], operator.add]
    search_results: Annotated[List[dict], operator.add]
    answer: Optional[str]
    is_satisfactory: Optional[bool]
    reflection_feedback: Optional[str]
    iteration: int
    max_iterations: int
    searched_count: int  # 已搜索过的 query 数量，避免重复搜索

# ---- LLM ----
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ---- Tool ----
def search_web(query: str, num_results: int = 5) -> List[dict]:
    """返回搜索结果列表，每条包含 title, url, snippet"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
        return [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in results
        ]

# ---- Prompts ----
INITIAL_QUERY_PROMPT = """你是一个研究助理。请为以下问题生成2-3个精准的搜索查询，用换行分隔：
{question}"""

REFINE_QUERY_PROMPT = """你是一个研究助理，已经进行了几轮搜索但答案不够满意。
原始问题：{question}
已使用的查询：{queries}
当前答案缺陷：{feedback}
请提出2-3个新的搜索查询，弥补上述缺陷，用换行分隔。"""

ANSWER_PROMPT = """你是专业研究助理。请根据以下搜索结果，回答用户问题。要求：
1. 内容准确、结构清晰
2. 必须为每个事实标注引用编号（如 [1]）
3. 如果信息不足，请明确指出不足，不要编造

问题：{question}
搜索结果：
{context}
回答："""

REFLECT_PROMPT = """作为严苛的评审，请评估以下答案。判断它是否充分、准确地回答了原始问题。

原始问题：{question}
待评估答案：{answer}

评估标准：
- 是否直接回答了问题的所有部分？
- 是否有足够的证据支持（引用是否恰当）？
- 是否存在逻辑漏洞或事实错误？

请以JSON格式输出：
{{
  "is_satisfactory": true/false,
  "feedback": "如果不满意，给出改进方向（例如：缺少XX信息，需要搜索XX）"
}}"""

# ---- Nodes ----
def generate_query(state: ResearchState) -> dict:
    if not state["queries"]:
        prompt = INITIAL_QUERY_PROMPT.format(question=state["question"])
    else:
        feedback = state.get("reflection_feedback") or "信息不足"
        prompt = REFINE_QUERY_PROMPT.format(
            question=state["question"],
            queries=state["queries"],
            feedback=feedback,
        )
    response = llm.invoke(prompt)
    new_queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    return {"queries": new_queries, "iteration": state["iteration"] + 1}

def search(state: ResearchState) -> dict:
    searched = state.get("searched_count", 0)
    new_queries = state["queries"][searched:]  # 只搜索本轮新增的 query
    new_results = []
    for q in new_queries:
        try:
            new_results.extend(search_web(q))
        except Exception:
            continue
    return {
        "search_results": new_results,
        "searched_count": len(state["queries"]),
    }

def generate_answer(state: ResearchState) -> dict:
    context_parts = []
    for i, res in enumerate(state["search_results"]):
        context_parts.append(
            f"[{i+1}] {res['title']}\n{res['snippet']}\n来源: {res['url']}\n"
        )
    context = "\n".join(context_parts)
    prompt = ANSWER_PROMPT.format(question=state["question"], context=context)
    response = llm.invoke(prompt)
    return {"answer": response.content}

def reflect(state: ResearchState) -> dict:
    prompt = REFLECT_PROMPT.format(
        question=state["question"], answer=state["answer"]
    )
    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, KeyError):
        result = {"is_satisfactory": False, "feedback": "无法解析评估结果，请重新搜索"}
    return {
        "is_satisfactory": result.get("is_satisfactory", False),
        "reflection_feedback": result.get("feedback", ""),
    }

# ---- Graph ----
workflow = StateGraph(ResearchState)

workflow.add_node("generate_query", generate_query)
workflow.add_node("search", search)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("reflect", reflect)

workflow.set_entry_point("generate_query")
workflow.add_edge("generate_query", "search")
workflow.add_edge("search", "generate_answer")
workflow.add_edge("generate_answer", "reflect")

def should_continue(state: ResearchState) -> str:
    if state["is_satisfactory"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    return "generate_query"

workflow.add_conditional_edges(
    "reflect",
    should_continue,
    {"generate_query": "generate_query", "end": END},
)

app = workflow.compile()

# ---- Entry Point ----
if __name__ == "__main__":
    initial_state: ResearchState = {
        "question": "2026年大模型Agent领域最重要的三篇论文是什么？它们的核心贡献分别是什么？",
        "queries": [],
        "search_results": [],
        "answer": None,
        "is_satisfactory": False,
        "reflection_feedback": None,
        "iteration": 0,
        "max_iterations": 3,
        "searched_count": 0,
    }
    final_state = app.invoke(initial_state)

    print("最终答案：")
    print(final_state["answer"])
    print("\n搜索历史：", final_state["queries"])
