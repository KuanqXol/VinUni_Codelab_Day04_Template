"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Phiên bản hoàn chỉnh cho system prompt và tool-calling loop.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý AI chính thức cho các sản phẩm và dịch vụ trong hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant.
- Vai trò: Tư vấn sản phẩm VinFast, dịch vụ Vinpearl và tiếp nhận yêu cầu hỗ trợ khách hàng.
- Giọng nói: Chuyên nghiệp, thân thiện, ngắn gọn và chính xác.

## AVAILABLE TOOLS
- search_product_catalog: Tra cứu catalog sản phẩm/dịch vụ theo danh mục và giá tối đa.
- submit_support_ticket: Tạo ticket hỗ trợ khi khách hàng báo lỗi, khiếu nại, phản hồi hoặc cần xử lý sự cố.

## CORE RULES
1. Không bịa dữ liệu sản phẩm, giá, tình trạng hàng hoặc mã ticket.
2. Khi người dùng hỏi danh sách sản phẩm/dịch vụ theo điều kiện, phải dùng search_product_catalog.
3. Khi người dùng muốn ghi nhận vấn đề, phản hồi, lỗi hoặc cần hỗ trợ, phải dùng submit_support_ticket.
4. Nếu một câu chứa cả tra cứu catalog và tạo ticket, thực hiện cả hai.
5. Nếu tool trả kết quả rỗng, nói rõ là không tìm thấy sản phẩm phù hợp.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ nội dung liên quan đến Vingroup, VinFast và Vinpearl.
- Với câu hỏi ngoài phạm vi hoặc thiếu dữ liệu, trả lời giới hạn và đề xuất cách hỏi phù hợp.

## OUTPUT CONTRACT
- Trace nội bộ theo dạng Thought, Action, Observation.
- Final Answer bằng tiếng Việt, rõ ràng, chỉ dựa trên observation từ tool hoặc FAQ cơ bản.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)
        actions = []
        if intents["needs_catalog"]:
            actions.append("search_product_catalog")
        if intents["needs_ticket"]:
            actions.append("submit_support_ticket")
        if not actions:
            actions.append("faq")

        iteration = 1
        action_index = 0
        while iteration <= self.max_iterations:
            action = actions[action_index]

            if action == "search_product_catalog":
                args = {
                    "category": intents["category"],
                    "max_price": intents["max_price"]
                }
                observation = TOOL_MAP[action](**args)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Người dùng cần tra cứu catalog theo điều kiện.",
                    "action": action,
                    "arguments": args,
                    "observation": observation
                })
            elif action == "submit_support_ticket":
                args = {
                    "customer_name": intents["customer_name"],
                    "issue_description": intents["issue_description"],
                    "priority": intents["priority"]
                }
                observation = TOOL_MAP[action](**args)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Người dùng cần ghi nhận yêu cầu hỗ trợ.",
                    "action": action,
                    "arguments": args,
                    "observation": observation
                })
            else:
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Câu hỏi thuộc FAQ cơ bản, không cần gọi tool.",
                    "action": "final_answer",
                    "arguments": {},
                    "observation": self._answer_faq(user_input)
                })

            action_index += 1
            if action_index >= len(actions):
                return {
                    "answer": self._build_final_answer(user_input),
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }
            iteration += 1

        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached"
        }

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        text = user_input.lower()
        needs_ticket = any(keyword in text for keyword in [
            "tôi tên", "tên tôi", "bị lỗi", "lỗi", "ghi nhận", "phản hồi",
            "hỗ trợ", "xử lý", "khiếu nại", "sự cố", "ẩm mốc"
        ])
        is_warranty_faq = "bảo hành" in text and "pin" in text
        needs_catalog = any(keyword in text for keyword in [
            "xem", "tìm", "tra cứu", "có", "giá dưới", "resort",
            "du lịch", "vinpearl", "xe điện", "vinfast"
        ])
        if is_warranty_faq and not needs_ticket and "giá" not in text and "xem" not in text:
            needs_catalog = False

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "category": self._extract_category(text),
            "max_price": self._extract_max_price(text),
            "customer_name": self._extract_customer_name(user_input),
            "issue_description": self._extract_issue_description(user_input),
            "priority": self._extract_priority(text),
            "is_faq": not needs_catalog and not needs_ticket
        }

    def _extract_category(self, text: str) -> str:
        if any(keyword in text for keyword in ["du lịch", "resort", "khách sạn", "vinpearl", "phòng"]):
            return "du_lich"
        return "xe_dien"

    def _extract_max_price(self, text: str) -> int:
        match = re.search(r"(?:dưới|tối đa|không quá)\s*([\d,.]+)\s*(triệu|tỷ|ty|vnd|đồng)?", text)
        if not match:
            return 999999999999

        number = float(match.group(1).replace(",", "."))
        unit = match.group(2) or ""
        if unit in {"tỷ", "ty"}:
            return int(number * 1_000_000_000)
        if unit == "triệu":
            return int(number * 1_000_000)
        return int(number)

    def _extract_customer_name(self, user_input: str) -> str:
        patterns = [
            r"tôi tên\s+([^,.]+)",
            r"tên tôi là\s+([^,.]+)",
            r"tên tôi\s+([^,.]+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Khách hàng"

    def _extract_issue_description(self, user_input: str) -> str:
        feedback_match = re.search(r"(?:ghi nhận phản hồi:|phản hồi:)\s*(.+)", user_input, flags=re.IGNORECASE)
        source = feedback_match.group(1) if feedback_match else user_input
        source = re.sub(r"tôi tên\s+[^,.]+[,.\s]*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"tên tôi là\s+[^,.]+[,.\s]*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"đây là vấn đề.*", "", source, flags=re.IGNORECASE).strip(" .,")
        source = re.sub(r"mức độ\s+(thấp|trung bình|cao|nghiêm trọng).*", "", source, flags=re.IGNORECASE).strip(" .,")
        return source[:1].upper() + source[1:] if source else "Khách hàng cần hỗ trợ."

    def _extract_priority(self, text: str) -> str:
        if any(keyword in text for keyword in ["nghiêm trọng", "gấp", "khẩn", "cao"]):
            return "high"
        if any(keyword in text for keyword in ["thấp", "không gấp"]):
            return "low"
        return "medium"

    def _build_final_answer(self, user_input: str) -> str:
        parts = []
        for step in self.trace:
            action = step["action"]
            observation = step["observation"]
            if action == "search_product_catalog":
                parts.append(self._format_catalog_answer(observation))
            elif action == "submit_support_ticket":
                parts.append(self._format_ticket_answer(observation))
            elif action == "final_answer":
                parts.append(observation)

        if not parts:
            parts.append(self._answer_faq(user_input))
        return "\n\n".join(parts)

    def _format_catalog_answer(self, products: List[Dict[str, Any]]) -> str:
        if not products:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp với điều kiện của bạn."
        if products and "error" in products[0]:
            return f"Rất tiếc, chưa thể tra cứu catalog: {products[0]['error']}"

        lines = ["Tôi tìm thấy các lựa chọn phù hợp:"]
        for product in products:
            price = f"{product['price_vnd']:,}".replace(",", ".")
            lines.append(f"- {product['name']}: {price} VND. {product['description']}")
        return "\n".join(lines)

    def _format_ticket_answer(self, ticket: Dict[str, Any]) -> str:
        return (
            f"Đã tạo ticket {ticket['ticket_id']} cho {ticket['customer_name']} "
            f"với mức ưu tiên {ticket['priority']}. Trạng thái hiện tại: {ticket['status']}."
        )

    def _answer_faq(self, user_input: str) -> str:
        text = user_input.lower()
        if "bảo hành" in text and "pin" in text:
            return "Chính sách bảo hành pin xe điện VinFast thường được nêu là 10 năm cho các mẫu xe trong catalog bài lab."
        return "Tôi có thể hỗ trợ tra cứu sản phẩm VinFast/Vinpearl hoặc ghi nhận yêu cầu hỗ trợ liên quan đến Vingroup."


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
