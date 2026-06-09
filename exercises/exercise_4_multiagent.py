"""Bài Tập 4: Thêm Privacy Agent vào Multi-Agent System."""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: giá trị mới ghi đè giá trị cũ."""
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    final_response: str


async def call_llm_with_fallback(prompt: str, fallback: str, timeout_seconds: int = 25) -> str:
    """Call the LLM, but keep the exercise runnable when the API is slow."""
    try:
        llm = get_llm()
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=timeout_seconds,
        )
        return response.content
    except Exception as exc:
        return f"{fallback}\n\n(Lưu ý: dùng fallback vì LLM/API phản hồi lỗi hoặc quá chậm: {exc})"


async def law_agent(state: State) -> dict:
    """Agent phân tích pháp lý tổng quát."""
    prompt = f"""Bạn là chuyên gia pháp lý. Phân tích câu hỏi sau:

{state['question']}

Tập trung vào: hợp đồng, trách nhiệm dân sự, quyền và nghĩa vụ pháp lý."""
    
    fallback = (
        "Có thể phát sinh trách nhiệm dân sự, nghĩa vụ thông báo cho khách hàng/cơ quan có thẩm quyền, "
        "bồi thường thiệt hại nếu chứng minh được tổn thất, và rủi ro vi phạm hợp đồng hoặc điều khoản bảo mật."
    )
    return {"law_analysis": await call_llm_with_fallback(prompt, fallback)}


def check_routing(state: State) -> list[Send]:
    """Quyết định gọi agents nào dựa trên nội dung câu hỏi."""
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]


async def tax_agent(state: State) -> dict:
    """Agent chuyên về thuế."""
    prompt = f"""Bạn là chuyên gia thuế. Phân tích khía cạnh thuế trong câu hỏi:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: IRS, tax evasion, penalties, FBAR, FATCA."""
    
    fallback = (
        "Nếu sự việc có yếu tố thuế, công ty cần rà soát nghĩa vụ khai báo, nộp thuế, tiền phạt chậm nộp "
        "và rủi ro kiểm tra/điều tra bởi cơ quan thuế. Nếu không có dữ kiện thuế cụ thể, cần tách riêng khỏi sự cố dữ liệu."
    )
    return {"tax_analysis": await call_llm_with_fallback(prompt, fallback)}


async def compliance_agent(state: State) -> dict:
    """Agent chuyên về compliance."""
    prompt = f"""Bạn là chuyên gia compliance. Phân tích khía cạnh tuân thủ:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: SEC, SOX, FCPA, AML, regulatory violations."""
    
    fallback = (
        "Cần đánh giá nghĩa vụ tuân thủ nội bộ, lưu vết xử lý sự cố, báo cáo cho cơ quan quản lý nếu luật áp dụng yêu cầu, "
        "và cập nhật chính sách bảo mật, kiểm soát truy cập, đào tạo nhân sự."
    )
    return {"compliance_analysis": await call_llm_with_fallback(prompt, fallback)}


async def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: GDPR, data protection, privacy rights, consent, data breach, CCPA."""
    
    fallback = (
        "Sự cố rò rỉ dữ liệu có thể kích hoạt nghĩa vụ thông báo vi phạm, điều tra nguyên nhân, giảm thiểu thiệt hại, "
        "đáp ứng quyền của chủ thể dữ liệu, và chịu phạt theo GDPR/CCPA hoặc luật bảo vệ dữ liệu liên quan."
    )
    return {"privacy_analysis": await call_llm_with_fallback(prompt, fallback)}


async def aggregate_results(state: State) -> dict:
    """Tổng hợp kết quả từ tất cả agents."""
    sections = []
    if state.get("law_analysis"):
        sections.append(f"📋 PHÂN TÍCH PHÁP LÝ:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"💰 PHÂN TÍCH THUẾ:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"✅ PHÂN TÍCH TUÂN THỦ:\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"🔒 PHÂN TÍCH QUYỀN RIÊNG TƯ:\n{state['privacy_analysis']}")
    
    combined = "\n\n".join(sections)
    
    prompt = f"""Tổng hợp các phân tích sau thành một báo cáo pháp lý hoàn chỉnh:

{combined}

Câu hỏi gốc: {state['question']}

Hãy tạo một báo cáo ngắn gọn, có cấu trúc rõ ràng."""
    
    fallback = (
        f"{combined}\n\nKẾT LUẬN:\n"
        "Công ty nên xử lý sự cố theo hướng: cô lập nguyên nhân, thông báo khi luật yêu cầu, "
        "đánh giá nghĩa vụ thuế/tuân thủ liên quan, lưu hồ sơ xử lý, và tham vấn luật sư trước khi phản hồi chính thức."
    )
    return {"final_response": await call_llm_with_fallback(prompt, fallback)}


def build_graph() -> StateGraph:
    """Xây dựng multi-agent graph."""
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("law_agent", law_agent)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)
    
    # Define edges
    graph.add_edge(START, "law_agent")
    graph.add_conditional_edges("law_agent", check_routing)
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)
    
    return graph.compile()


async def main():
    load_dotenv()
    
    # Test với câu hỏi có liên quan đến privacy
    question = "Nếu công ty bị rò rỉ dữ liệu khách hàng, hậu quả pháp lý và thuế là gì?"
    
    print("=" * 70)
    print("MULTI-AGENT SYSTEM với Privacy Agent")
    print("=" * 70)
    print(f"\nCâu hỏi: {question}\n")
    print("Đang xử lý qua các agents...\n")
    
    graph = build_graph()
    
    result = await graph.ainvoke({
        "question": question,
        "law_analysis": "",
        "tax_analysis": "",
        "compliance_analysis": "",
        "privacy_analysis": "",
        "final_response": "",
    })
    
    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 70)
    print(result["final_response"])
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
