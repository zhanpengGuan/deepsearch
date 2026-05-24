import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Annotated, List, Optional
import operator

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import requests

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
llm_fast = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ---- Tool ----
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def search_web(query: str, num_results: int = 5) -> List[dict]:
    """通过 Serper (Google) 搜索，返回 title, url, snippet"""
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": num_results},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r["title"], "url": r["link"], "snippet": r.get("snippet", "")}
        for r in data.get("organic", [])
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
    t0 = time.perf_counter()
    print(f"\n{'='*50}")
    print(f"[generate_query] 第 {state['iteration']+1} 轮 · 生成搜索关键词...")
    if not state["queries"]:
        prompt = INITIAL_QUERY_PROMPT.format(question=state["question"])
    else:
        feedback = state.get("reflection_feedback") or "信息不足"
        prompt = REFINE_QUERY_PROMPT.format(
            question=state["question"],
            queries=state["queries"],
            feedback=feedback,
        )
    response = llm_fast.invoke(prompt)
    new_queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    print(f"  新 query: {new_queries}")
    print(f"  耗时: {time.perf_counter() - t0:.1f}s")
    return {"queries": new_queries, "iteration": state["iteration"] + 1}

def search(state: ResearchState) -> dict:
    t0 = time.perf_counter()
    searched = state.get("searched_count", 0)
    new_queries = state["queries"][searched:]

    print(f"[search] 搜索 {len(new_queries)} 个 query (并行)...")
    new_results = []
    with ThreadPoolExecutor(max_workers=len(new_queries)) as executor:
        futures = {executor.submit(search_web, q): q for q in new_queries}
        for future in as_completed(futures):
            try:
                new_results.extend(future.result())
            except Exception as e:
                print(f"  [warn] 搜索失败: {futures[future]} — {e}")
    print(f"  获取 {len(new_results)} 条结果")
    print(f"  耗时: {time.perf_counter() - t0:.1f}s")
    return {
        "search_results": new_results,
        "searched_count": len(state["queries"]),
    }

def generate_answer(state: ResearchState) -> dict:
    t0 = time.perf_counter()
    print(f"[generate_answer] 基于 {len(state['search_results'])} 条搜索结果生成答案...")
    context_parts = []
    for i, res in enumerate(state["search_results"]):
        context_parts.append(
            f"[{i+1}] {res['title']}\n{res['snippet']}\n来源: {res['url']}\n"
        )
    context = "\n".join(context_parts)
    prompt = ANSWER_PROMPT.format(question=state["question"], context=context)
    response = llm.invoke(prompt)
    print(f"  耗时: {time.perf_counter() - t0:.1f}s")
    return {"answer": response.content}

def reflect(state: ResearchState) -> dict:
    t0 = time.perf_counter()
    print(f"[reflect] 评估答案...")
    prompt = REFLECT_PROMPT.format(
        question=state["question"], answer=state["answer"]
    )
    response = llm_fast.invoke(prompt)
    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, KeyError):
        result = {"is_satisfactory": False, "feedback": "无法解析评估结果，请重新搜索"}
    is_satisfied = result.get("is_satisfactory", False)
    status = "通过" if is_satisfied else "需改进"
    print(f"  评估结果: {status}")
    print(f"  耗时: {time.perf_counter() - t0:.1f}s")
    return {
        "is_satisfactory": is_satisfied,
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

    print(f"问题: {initial_state['question']}")
    t_start = time.perf_counter()

    final_state = app.invoke(initial_state)

    print(f"\n{'='*50}")
    print(f"总耗时: {time.perf_counter() - t_start:.1f}s")
    print(f"迭代次数: {final_state['iteration']}")
    print(f"最终答案：")
    print(final_state["answer"])
    print("\n搜索历史：", final_state["queries"])
