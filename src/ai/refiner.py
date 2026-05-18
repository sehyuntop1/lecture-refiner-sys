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


async def map_and_refine_script(slide_texts: list[str], raw_script: str, lecture_info: str) -> tuple[str, dict]:
    """슬라이드 한 장씩 대본 매핑 + 정제 동시 수행"""
    results = []
    total_input = 0
    total_output = 0

    for i, slide_text in enumerate(slide_texts):
        page_num = i + 1
        prompt = f"""
당신은 의학과 강의 슬라이드와 대본을 매핑하고 정제하는 전문가입니다.

아래는 {lecture_info} 강의의 슬라이드 {page_num}페이지 내용입니다.
전체 강의 대본에서 이 슬라이드를 설명하는 부분을 찾아서 발췌하되, 동시에 아래 정제 조건에 따라 다듬어주세요.

[슬라이드 {page_num} 내용]
{slide_text}

[전체 강의 대본]
{raw_script}

[정제 조건]
1. 오탈자를 수정하되, 의학 용어에 맞게 수정 (예: 미토콘돌아 → 미토콘드리아)
2. 교수님이 학생에게 질문한 경우 대화 형식 유지 (교수님: 질문 / 학생: 답변)
3. 불필요한 미사여구 제거, 구어체는 반드시 유지할 것
4. 교수님 농담 살리기
5. 의학용어는 영어로 + 괄호 안에 한국어 번역 (예: smallpox(천연두))
6. 절대 요약하지 말 것. 산문 형식 유지
7. 강조하거나 기억하라고 한 내용 반드시 살리기
8. 관련 내용이 없으면 "해당 없음" 출력

정제된 발췌 내용만 출력하세요. 다른 설명 없이 바로 본문만 출력하세요.
"""
        text, cost = await _generate(prompt)
        total_input += cost["input_tokens"]
        total_output += cost["output_tokens"]
        results.append(f"[슬라이드 {page_num}]\n{text.strip()}")

    final_text = "\n\n".join(results)
    total_cost = calculate_cost(total_input, total_output)
    return final_text, total_cost


async def extract_emphasis(raw_script: str, lecture_info: str) -> tuple[str, dict]:
    prompt = f"""
당신은 의학과 시험 준비를 돕는 전문가입니다.
아래는 {lecture_info} 강의 대본입니다.

다음 기준에 따라 시험에 출제될 가능성이 높은 의학적 내용만 엄격하게 발췌해 주세요.

[발췌 기준]
- 교수님이 "시험에 나온다", "외워라", "반드시 알아야 한다", "중요하다", "기억해라" 등 명시적으로 강조한 의학 내용
- 교수님이 특정 수치, 메커니즘, 질환명, 용어를 반복해서 강조한 내용
- 교수님이 "이것만큼은", "핵심은", "포인트는" 등으로 콕 집어 언급한 내용

[절대 포함하지 말 것]
- AI, 기술 사용법, 수업 운영 방식에 대한 당부
- 일상적인 공지사항, 출석, 과제 관련 내용
- 단순한 예시나 비유로 든 내용 (강조 없이)
- 교수님 농담이나 잡담

각 항목은 아래 형식으로 작성하세요:
[중요] 발췌한 의학 내용

원본 대본:
{raw_script}

중요 의학 내용 발췌본만 출력하세요.
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
