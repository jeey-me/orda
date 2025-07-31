import openai
import json
from typing import List, Dict

# ✅ OpenAI SDK v1 방식 클라이언트
client = openai.OpenAI()

def build_prompt(issue_text: str, candidates: List[str], mode: str = "industry") -> str:
    task_type = {
        "industry": "관련 산업을 선택",
        "past": "관련 과거 이슈를 선택"
    }.get(mode, "관련 항목을 선택")

    prompt = f"""
다음은 현재 뉴스 이슈입니다:

[현재 이슈]
{issue_text}

아래는 {task_type}할 수 있는 후보 리스트입니다:

[후보 리스트]
{chr(10).join(f"- {cand}" for cand in candidates)}

당신은 전문 시장분석 애널리스트입니다. 초보자들에게 현재 이슈에 대해 어떠한 관련 산업과 과거 이슈가 관련되어 있는지
전문적이면서 친절하게 알려주는 역할을 수행해야합니다. 당신의 과제는 이슈와 가장 밀접한 후보를 관련성 순으로 정렬하고, 각각의 이유를 설명하는 것입니다.
응답은 JSON 형식으로 출력해주세요. 각 항목은 name과 reason 필드를 반드시 포함해야 합니다.

[출력 예시]
[
  {{
    "name": "반도체",
    "reason": "반도체의중요성은2018년이후미.중패권전쟁으로한층더강화되었다. 미.중패권전쟁은 처음에는무역전쟁으로시작되어현재는완전히기술패권전쟁으로전환되었는데, 그기술패권 전쟁의핵심대상이바로‘반도체’다."
  }},
  {{
    "name": "전자부품",
    "reason": "앞으로 전자 부품 시장은 전자 기기에 대한 수요 증가에 힘입어 연간 성장 궤적을 계속할 것으로 예상됩니다. 그러나 공급망 중단, 가격 변동성, 규제 준수와 같은 일상적인 도전 과제가 시장 환경을 계속해서 형성할 것입니다. 공급망 회복력을 강화하고 위험을 완화하기 위한 노력이 탄력을 받을 것으로 예상되며, 구매자는 단기적인 도전을 충족하기 위해 공급망 가시성, 다양화 및 협력 이니셔티브에 참여할 것입니다."
  }}
]
"""
    return prompt.strip()

def rerank_with_llm(issue_text: str, candidates: List[str], mode: str = "industry") -> List[Dict[str, str]]:
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "당신은 금융 투자 분석 전문가입니다."},
            {"role": "user", "content": build_prompt(issue_text, candidates, mode=mode)}
        ],
        temperature=0.3
    )

    output_text = response.choices[0].message.content.strip()
    try:
        return json.loads(output_text)
    except Exception as e:
        print("⚠️ JSON 파싱 실패. 응답 원문:", output_text)
        raise e