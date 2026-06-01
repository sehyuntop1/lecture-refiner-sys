# -*- coding: utf-8 -*-
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
                model="gemini-2.5-flash-lite",
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
                model="gemini-2.5-flash-lite",
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


def _split_script_into_sentences(raw_script: str) -> list[str]:
    """대본을 문장 단위로 나누되, 교수/학생 발화 기준으로 분리"""
    # 줄바꿈 기준으로 우선 분리한 뒤 너무 짧은 줄은 앞에 합침
    lines = [l.strip() for l in raw_script.split("\n") if l.strip()]
    merged = []
    buf = ""
    for line in lines:
        if len(line) < 15 and buf:
            buf += " " + line
        else:
            if buf:
                merged.append(buf)
            buf = line
    if buf:
        merged.append(buf)
    return merged


async def split_script_by_slides(
    slide_texts: list[str], raw_script: str, lecture_info: str
) -> tuple[list[str], dict]:
    """
    개선된 맵핑 전략:
    1. 전체 슬라이드 텍스트를 온전히 제공 (앞 100자 자르기 제거)
    2. CHUNK 단위로 대본을 나눠서 각 청크마다 해당 슬라이드 구간 지정
    3. 슬라이드 순서 단조증가 제약을 강제
    4. 건너뛴 슬라이드 명시 요청 강화
    """
    total_pages = len(slide_texts)
    total_input = 0
    total_output = 0

    # ── 슬라이드 목록 (전체 텍스트 제공, 단 300자 이내로 요약)
    slide_list_lines = []
    for i, text in enumerate(slide_texts):
        # 줄바꿈 제거 후 300자
        one_line = " ".join(text.split())[:300]
        slide_list_lines.append(f"[슬라이드 {i+1}] {one_line}")
    full_slide_list = "\n".join(slide_list_lines)

    # ── 대본을 ~600자 청크로 분할 (문장 경계 존중)
    sentences = _split_script_into_sentences(raw_script)
    chunks: list[str] = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) > 600 and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sent
        else:
            current_chunk = (current_chunk + "\n" + sent).strip()
    if current_chunk:
        chunks.append(current_chunk.strip())

    total_chunks = len(chunks)

    # ── 각 청크를 담당 슬라이드에 배정
    # 단조증가 제약: 이전 청크의 최대 슬라이드 번호 이상만 허용
    chunk_assignments: list[int | None] = []   # None = 해당없음
    prev_max_slide = 0

    ASSIGN_BATCH = 5  # 한 번에 몇 청크씩 처리할지
    for batch_start in range(0, total_chunks, ASSIGN_BATCH):
        batch_chunks = chunks[batch_start: batch_start + ASSIGN_BATCH]
        batch_numbered = "\n\n".join([
            f"[청크 {batch_start + j + 1}]\n{c}"
            for j, c in enumerate(batch_chunks)
        ])

        prompt = f"""당신은 의학 강의 대본과 슬라이드를 정밀 매핑하는 전문가입니다.

강의: {lecture_info}
총 슬라이드 수: {total_pages}
현재까지 처리된 슬라이드 범위: 슬라이드 1 ~ {prev_max_slide} 은 이미 이전 청크에서 사용됨.

[전체 슬라이드 목록]
{full_slide_list}

[이번에 배정할 대본 청크들]
{batch_numbered}

[배정 규칙 - 엄격히 준수]
1. 각 청크가 어느 슬라이드를 설명하는지 판단하세요.
2. 슬라이드 번호는 반드시 단조증가해야 합니다. (앞 청크 슬라이드 번호 <= 현재 청크)
3. 이전에 사용된 슬라이드 1~{prev_max_slide}보다 이전 번호는 절대 사용 불가.
4. 청크 내용이 슬라이드 내용과 전혀 관련 없으면 slide_no를 null로 하세요.
5. 교수님이 슬라이드를 설명하지 않고 건너뛴 경우 해당 슬라이드 번호는 건너뜁니다 (결과에 포함 안 해도 됨).
6. 여러 청크가 같은 슬라이드에 해당할 수 있습니다.

반드시 아래 JSON만 출력하세요:
{{
  "assignments": [
    {{"chunk": {batch_start + 1}, "slide_no": 3}},
    {{"chunk": {batch_start + 2}, "slide_no": 3}},
    {{"chunk": {batch_start + 3}, "slide_no": null}}
  ]
}}
"""
        text, cost = await _generate(prompt)
        total_input += cost["input_tokens"]
        total_output += cost["output_tokens"]

        try:
            clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
            json_match = re.search(r"\{.*\}", clean, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data.get("assignments", []):
                    sn = item.get("slide_no")
                    if sn is not None:
                        sn = max(int(sn), prev_max_slide)  # 단조증가 강제
                        sn = min(sn, total_pages)
                        prev_max_slide = sn
                    chunk_assignments.append(sn)
            else:
                chunk_assignments.extend([None] * len(batch_chunks))
        except Exception:
            chunk_assignments.extend([None] * len(batch_chunks))

    # ── 슬라이드별로 청크 합치기
    slide_chunks: dict[int, list[str]] = {i: [] for i in range(1, total_pages + 1)}
    for idx, slide_no in enumerate(chunk_assignments):
        if slide_no is not None and 1 <= slide_no <= total_pages:
            slide_chunks[slide_no].append(chunks[idx])

    page_scripts = []
    for i in range(1, total_pages + 1):
        if slide_chunks[i]:
            page_scripts.append("\n".join(slide_chunks[i]))
        else:
            page_scripts.append("해당 없음")

    return page_scripts, calculate_cost(total_input, total_output)


async def refine_page_scripts(page_scripts: list[str], lecture_info: str) -> tuple[list[str], dict]:
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
1. 음성인식 오류로 잘못 표기된 의학 용어를 정확한 영문 표기로 반드시 교정하세요.
   - 한국어 발음으로 쓰인 의학 용어를 영문 정식 표기로 변환
   - 예시:
     파리에탈셀 / 파리텔셀 / 파리에탈 → parietal cell(벽세포)
     프로톤 펌프 → proton pump(양성자 펌프)
     피켓 / 피캡 / 피캅 → vonoprazan(P-CAB)
     가스트릭 얼서 / 가스트릭 월서 → gastric ulcer(위궤양)
     에비던스 → evidence(근거)
     가이드라인 → guideline(진료 지침)
     인액티브 / 인 액티브 → inactive form(비활성형)
     액티브 → active form(활성형)
     뮤코사 → mucosa(점막)
     헬리코박터 → Helicobacter pylori(헬리코박터 파일로리)
   - 이 외에도 문맥상 명확히 특정 의학 용어를 발음한 것으로 판단되면 적극적으로 교정할 것
   - 확실하지 않은 경우 [?원문표기] 형식으로 표시
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
