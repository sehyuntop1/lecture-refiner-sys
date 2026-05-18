import json
import re
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

INPUT_PRICE_PER_1M = 0.30
OUTPUT_PRICE_PER_1M = 2.50
KRW_RATE = 1350


def calculate_cost(input_tokens: int, output_tokens: int) -> dict:
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M * KRW_RATE
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M * KRW_RATE
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_krw": input_cost + output_cost,
    }


async def _generate(prompt: str) -> tuple[str, dict]:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(temperature=0.0),
    )
    response = await model.generate_content_async(prompt)
    usage = response.usage_metadata
    cost = calculate_cost(usage.prompt_token_count, usage.candidates_token_count)
    return response.text, cost


async def split_script_by_slides(slide_texts: list[str], raw_script: str, lecture_info: str) -> tuple[list[str], dict]:
    """1단계: 전체 대본을 슬라이드별로 분할"""
    total_pages = len(slide_texts)
    slide_list = "\n".join([f"[슬라이드 {i+1}] {text[:200]}" for i, text in enumerate(slide_texts)])

    prompt = f"""당신은 의학과 강의 슬라이드와 대본을 매핑하는 전문가입니다.

아래는 {lecture_info} 강의입니다.
전체 강의 대본을 슬라이드 {total_pages}페이지에 맞게 분할해주세요.

[슬라이드 목록]
{slide_list}

[전체 강의 대본]
{raw_script}

반드시 아래 JSON 형식으로만 출력하세요:
{{
  "pages": [
    {{"page": 1, "script": "슬라이드 1에 해당하는 대본 내용"}},
    {{"page": 2, "script": "슬라이드 2에 해당하는 대본 내용"}}
  ]
}}

- 관련 내용 없으면 script를 "해당 없음"으로
- 전체 대본 내용이 빠지지 않게 분배
- JSON만 출력, 다른 설명 없이
"""
    text, cost = await _generate(prompt)

    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            pages = data.get("pages", [])
            result = {p["page"]: p.get("script", "해당 없음") for p in pages}
            return [result.get(i+1, "해당 없음") for i in range(total_pages)], cost
    except:
        pass

    return ["해당 없음"] * total_pages, cost


async def refine_page_scripts(page_scripts: list[str], lecture_info: str) -> tuple[list[str], dict]:
    """2단계: 분할된 대본 정제"""
    combined = "\n\n".join([f"[슬라이드 {i+1}]\n{script}" for i, script in enumerate(page_scripts)])

    prompt = f"""당신은 의학과 강의 대본을 정제하는 전문가입니다.
아래는 {lecture_info} 강의의 슬라이드별 대본입니다.
각 슬라이드 대본을 아래 조건에 따라 정제하되, [슬라이드 N] 형식은 반드시 유지하세요.

[정제 조건]
1. 오탈자 수정 (의학 용어 기준, 예: 미토콘돌아 → 미토콘드리아)
2. 교수님 질문-학생 답변 대화 형식 유지
3. 불필요한 미사여구 제거, 구어체는 반드시 유지
4. 교수님 농담 살리기
5. 의학용어 영어로 + 괄호 안에 한국어 (예: smallpox(천연두))
6. 절대 요약하지 말 것, 산문 형식 유지
7. 강조/기억하라고 한 내용 반드시 살리기
8. "해당 없음"인 슬라이드는 그대로 "해당 없음" 출력

[슬라이드별 대본]
{combined}

정제된 슬라이드별 대본만 출력하세요. [슬라이드 N] 형식 유지하면서 본문만 출력하세요.
"""
    text, cost = await _generate(prompt)

    # [슬라이드 N] 기준으로 파싱
    refined = []
    pattern = re.split(r'\[슬라이드 \d+\]', text)
    pattern = [p.strip() for p in pattern if p.strip()]

    if len(pattern) == len(page_scripts):
        refined = pattern
    else:
        refined = page_scripts  # 파싱 실패시 원본 사용

    return refined, cost


async def extract_emphasis(raw_script: str, lecture_info: str) -> tuple[str, dict]:
    prompt = f"""당신은 의학과 시험 준비를 돕는 전문가입니다.
아래는 {lecture_info} 강의 대본입니다.

시험에 출제될 가능성이 높은 의학적 내용만 엄격하게 발췌해 주세요.

[발췌 기준]
- "시험에 나온다", "외워라", "반드시 알아야 한다", "중요하다", "기억해라" 등 명시적으로 강조한 의학 내용
- 특정 수치, 메커니즘, 질환명, 용어를 반복 강조한 내용
- "이것만큼은", "핵심은", "포인트는" 등으로 콕 집어 언급한 내용

[포함하지 말 것]
- AI, 기술 사용법, 수업 운영 방식
- 공지사항, 출석, 과제
- 강조 없는 단순 예시/비유
- 농담, 잡담

형식: [중요] 발췌한 의학 내용

원본 대본:
{raw_script}

발췌 내용만 출력하세요.
"""
    return await _generate(prompt)


async def extract_slide_texts_from_pdf(pdf_path: str) -> list[str]:
    import fitz
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text().strip()
        texts.append(text if text else "(텍스트 없음)")
    doc.close()
    return texts
