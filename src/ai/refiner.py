import json
import re
import asyncio
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


async def _generate(prompt: str, max_retries: int = 5) -> tuple[str, dict]:
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            usage = response.usage_metadata
            cost = calculate_cost(usage.prompt_token_count, usage.candidates_token_count)
            return response.text, cost

        except Exception as e:
            err_str = str(e)
            is_retryable = (
                "503" in err_str or
                "504" in err_str or
                "UNAVAILABLE" in err_str or
                "CANCELLED" in err_str or
                "429" in err_str
            )
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
                continue
            else:
                raise


async def _generate_with_image(img_bytes: bytes, prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            return response.text.strip()

        except Exception as e:
            err_str = str(e)
            is_retryable = (
                "503" in err_str or
                "504" in err_str or
                "UNAVAILABLE" in err_str or
                "CANCELLED" in err_str or
                "429" in err_str
            )
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
                continue
            else:
                raise


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
            text = await _generate_with_image(
                img_bytes,
                "이 강의 슬라이드에 있는 모든 텍스트를 그대로 추출해주세요. 텍스트만 출력하고 다른 설명은 하지 마세요."
            )
            texts.append(text)
        except Exception:
            texts.append(f"(페이지 {i+1} 추출 실패)")

    doc.close()
    return texts


async def split_script_by_slides(slide_texts: list[str], raw_script: str, lecture_info: str) -> tuple[list[str], dict]:
    """1단계: 전체 대본을 슬라이드별로 분할"""
    total_pages = len(slide_texts)
    slide_list = "\n".join([f"[슬라이드 {i+1}] {text[:100]}" for i, text in enumerate(slide_texts)])

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
        clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            pages = data.get("pages", [])
            result = {p["page"]: p.get("script", "해당 없음") for p in pages}
            return [result.get(i+1, "해당 없음") for i in range(total_pages)], cost
    except Exception:
        pass

    return ["해당 없음"] * total_pages, cost


async def refine_page_scripts(page_scripts: list[str], lecture_info: str) -> tuple[list[str], dict]:
    """2단계: 10장씩 나눠서 bullet 변환 후 합본"""
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

        prompt = f"""당신은 의학과 강의 대본을 정리하는 전문가입니다.
아래는 {lecture_info} 강의의 슬라이드별 대본입니다.
각 슬라이드 대본을 bullet 포인트 형식으로 변환하되, [슬라이드 N] 형식은 반드시 유지하세요.

[변환 규칙]
1. 구어체를 간결한 문어체 bullet 포인트(•)로 변환합니다.
2. 의학 내용은 절대 빠뜨리지 말고 전부 bullet으로 담아주세요.
3. bullet 하나당 하나의 개념/사실/흐름을 담습니다.
4. 교수님이 강조한 내용("기억하세요", "외워두세요", "중요합니다" 등)은 bullet 앞에 ★ 표시를 붙여주세요.
5. 공지사항, 출석, 과제, 잡담, 추임새는 제거합니다.
6. 의학용어는 영어 + 괄호 안에 한국어로 표기합니다. 예: bilirubin(빌리루빈)
7. 오탈자 및 잘못된 영어 표현은 올바른 의학 용어로 수정합니다.
8. "해당 없음"인 슬라이드는 그대로 "해당 없음" 출력합니다.

출력 형식 예시:
[슬라이드 1]
• bilirubin(빌리루빈)은 수용성이 아니므로 albumin(알부민)과 결합해 liver(간)로 이동
★ conjugation(포합) 후 bile(담즙)을 통해 intestine(장)으로 배출 — 반드시 기억
• intestine(장)에서 urobilinogen(유로빌리노겐)으로 전환

[슬라이드별 대본]
{combined}

변환된 슬라이드별 bullet 대본만 출력하세요. [슬라이드 N] 형식 반드시 유지하세요.
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
