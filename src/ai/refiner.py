import json
import re
import base64
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

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
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    usage = response.usage_metadata
    cost = calculate_cost(usage.prompt_token_count, usage.candidates_token_count)
    return response.text, cost


async def extract_slide_texts_from_pdf(pdf_path: str) -> list[str]:
    """PDF 슬라이드를 이미지로 변환 후 Gemini Vision으로 텍스트 추출"""
    import fitz

    doc = fitz.open(pdf_path)
    texts = []

    for i, page in enumerate(doc):
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    "이 강의 슬라이드에 있는 모든 텍스트를 그대로 추출해주세요. 텍스트만 출력하고 다른 설명은 하지 마세요."
                ],
            )
            texts.append(response.text.strip())
        except Exception:
            texts.append(f"(페이지 {i+1} 추출 실패)")

    doc.close()
    return texts


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
        # ```json ... ``` 블록 제거 후 파싱
        clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            pages = data.get("pages", [])
            result = {p["page"]: p.get("script", "해당 없음") for p in pages}
            return [result.get(i+1, "해당 없음") for i in range(total_pages)], cost
    except Exception as e:
        pass

    return ["해당 없음"] * total_pages, cost


async def refine_page_scripts(page_scripts: list[str], lecture_info: str) -> tuple[list[str], dict]:
    """2단계: 10장씩 나눠서 정제 후 합본"""
    BATCH_SIZE = 10
    all_refined = []
    total_input = 0
    total_output = 0

    for batch_start in range(0, len(page_scripts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(page_scripts))
        batch = page_scripts[batch_start:batch_end]

        combined = "\n\n".join([
            f"[슬라이드 {batch_start + i + 1}]\n{script}"
            for i, script in enumerate(batch)
        ])

        prompt = f"""당신은 의학과 강의 대본을 정제하는 전문가입니다.
아래는 {lecture_info} 강의의 슬라이드별 대본입니다.
각 슬라이드 대본을 아래 조건에 따라 정제하되, [슬라이드 N] 형식은 반드시 유지하세요.

[정제 조건]
1. 오탈자 수정 (의학 용어 기준, 예: 셀롤라 타입 → cellular type(세포 타입))
2. 콩글리시나 잘못된 영어 표현을 올바른 의학 영어로 수정
3. 교수님 질문-학생 답변 대화 형식 유지
4. 불필요한 미사여구 제거, 구어체는 반드시 유지
5. 교수님 농담은 살리기
6. 슬라이드 내용과 관련없는 교수님 사설, 개인적인 이야기, 주제 벗어난 잡담은 제거할 것 (단, 농담은 제외)
7. 의학용어는 영어로 + 괄호 안에 한국어 번역 (예: smallpox(천연두))
8. 절대 요약하지 말 것, 산문 형식 유지
9. 강조/기억하라고 한 내용 반드시 살리기
10. "해당 없음"인 슬라이드는 그대로 "해당 없음" 출력

[슬라이드별 대본]
{combined}

정제된 슬라이드별 대본만 출력하세요. [슬라이드 N] 형식 반드시 유지하면서 본문만 출력하세요.
"""
        text, cost = await _generate(prompt)
        total_input += cost["input_tokens"]
        total_output += cost["output_tokens"]

        parts = re.split(r'\[슬라이드 \d+\]', text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == len(batch):
            all_refined.extend(parts)
        else:
            all_refined.extend(batch)

    return all_refined, calculate_cost(total_input, total_output)


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
- 농담, 잡담, 사설

형식: [중요] 발췌한 의학 내용

원본 대본:
{raw_script}

발췌 내용만 출력하세요.
"""
    return await _generate(prompt)
