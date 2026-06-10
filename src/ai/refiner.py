# -*- coding: utf-8 -*-
import json
import re
import asyncio
import random
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


async def _generate(prompt: str, model: str = "gemini-2.5-flash", max_retries: int = 15) -> tuple[str, dict]:
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0),
                ),
                timeout=120.0  # 2분 타임아웃
            )
            usage = response.usage_metadata
            cost = calculate_cost(usage.prompt_token_count, usage.candidates_token_count)
            return response.text, cost
        except Exception as e:
            err_str = str(e)
            is_retryable = isinstance(e, asyncio.TimeoutError) or                            any(k in err_str for k in ["503", "504", "UNAVAILABLE", "CANCELLED", "429"]) or                            any(k in err_str.lower() for k in ["overloaded", "high demand"])
            if is_retryable and attempt < max_retries - 1:
                base_wait = min(2 ** attempt, 60)
                jitter = random.uniform(0, base_wait * 0.3)
                await asyncio.sleep(base_wait + jitter)
            else:
                raise


async def _generate_with_image(img_bytes: bytes, prompt: str, max_retries: int = 15) -> str:
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        prompt,
                    ],
                ),
                timeout=60.0  # 60초 타임아웃
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            is_retryable = isinstance(e, asyncio.TimeoutError) or                            any(k in err_str for k in ["503", "504", "UNAVAILABLE", "CANCELLED", "429"]) or                            any(k in err_str.lower() for k in ["overloaded", "high demand"])
            if is_retryable and attempt < max_retries - 1:
                base_wait = min(2 ** attempt, 30)
                jitter = random.uniform(0, base_wait * 0.3)
                await asyncio.sleep(base_wait + jitter)
            else:
                raise


async def extract_slide_texts_from_pdf(pdf_path: str) -> list[str]:
    """슬라이드별 텍스트 + 시각적 특징 추출"""
    import fitz

    doc = fitz.open(pdf_path)
    texts = []

    for i, page in enumerate(doc):
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        try:
            text = await _generate_with_image(
                img_bytes,
                """이 강의 슬라이드를 분석하세요.

[텍스트] 슬라이드의 모든 텍스트를 빠짐없이 추출
[시각] 슬라이드의 핵심 시각적 특징을 한 줄 요약 (이미지/다이어그램/표/사진 종류 및 주요 내용)
[유형] 아래 중 하나로 분류:
  - COVER: 표지 슬라이드
  - SECTION: 섹션 구분 슬라이드 (짧은 제목만 있는 구분용)
  - OBJECTIVES: 학습목표 슬라이드
  - CONTENT: 실제 강의 내용 슬라이드
  - IMAGE_ONLY: 이미지/사진만 있는 슬라이드
  - TABLE: 표/스코어링 시스템 슬라이드
  - SUMMARY: 요약/정리 슬라이드

텍스트가 없는 슬라이드도 반드시 [시각]과 [유형]을 작성하세요."""
            )
            texts.append(text)
        except Exception:
            texts.append(f"[텍스트] (페이지 {i+1} 추출 실패)\n[시각] 추출 실패\n[유형] CONTENT")

    doc.close()
    return texts


def _parse_slide_type(slide_text: str) -> str:
    """슬라이드 유형 파싱"""
    match = re.search(r'\[유형\]\s*(\w+)', slide_text)
    if match:
        return match.group(1).upper()
    return "CONTENT"


def _is_mappable_slide(slide_text: str) -> bool:
    """맵핑 대상 슬라이드인지 판단 (표지/섹션구분/학습목표 제외)"""
    slide_type = _parse_slide_type(slide_text)
    return slide_type not in ("COVER", "SECTION", "OBJECTIVES")


async def split_script_by_slides(
    slide_texts: list[str], raw_script: str, lecture_info: str
) -> tuple[list[str], list[str], list[int | None], dict]:
    """
    완전히 새로운 맵핑 전략:
    1. 슬라이드를 유형별로 분류 (표지/섹션구분 → 맵핑 제외)
    2. 대본 전체를 한 번에 AI에게 넘겨서 슬라이드별 대본 구간을 통째로 분할
    3. 대본이 너무 길면 절반씩 나눠서 처리 후 합침
    4. 검증 및 재시도 로직
    """
    total_pages = len(slide_texts)
    total_input = 0
    total_output = 0

    # ── 슬라이드 분류
    mappable_slides = []  # (원본 인덱스, 슬라이드 내용)
    for i, text in enumerate(slide_texts):
        if _is_mappable_slide(text):
            mappable_slides.append((i + 1, text))  # 1-indexed

    # ── 슬라이드 목록 구성 (맵핑 가능한 슬라이드만, 전체 텍스트)
    slide_list_lines = []
    for slide_no, text in mappable_slides:
        # [텍스트], [시각] 파싱
        text_match = re.search(r'\[텍스트\](.*?)(?=\[시각\]|\[유형\]|$)', text, re.DOTALL)
        visual_match = re.search(r'\[시각\](.*?)(?=\[유형\]|$)', text, re.DOTALL)
        slide_type = _parse_slide_type(text)

        extracted_text = text_match.group(1).strip()[:400] if text_match else " ".join(text.split())[:400]
        visual_desc = visual_match.group(1).strip()[:100] if visual_match else ""

        entry = f"[슬라이드 {slide_no}] ({slide_type})"
        if extracted_text:
            entry += f" 텍스트: {extracted_text}"
        if visual_desc:
            entry += f" | 시각: {visual_desc}"
        slide_list_lines.append(entry)

    full_slide_list = "\n".join(slide_list_lines)
    mappable_slide_numbers = [s[0] for s in mappable_slides]

    # ── 대본이 너무 길면 절반으로 나눠서 처리
    script_len = len(raw_script)
    SCRIPT_LIMIT = 15000  # 15000자 이상이면 분할

    if script_len <= SCRIPT_LIMIT:
        script_parts = [raw_script]
        slide_splits = [mappable_slide_numbers]
    else:
        # 절반 지점에서 슬라이드 목록도 절반 분할
        mid_script = script_len // 2
        # 줄바꿈 기준으로 자르기
        newline_pos = raw_script.rfind('\n', 0, mid_script)
        if newline_pos == -1:
            newline_pos = mid_script
        part1 = raw_script[:newline_pos]
        part2 = raw_script[newline_pos:]

        mid_slide = len(mappable_slide_numbers) // 2
        script_parts = [part1, part2]
        slide_splits = [mappable_slide_numbers[:mid_slide], mappable_slide_numbers[mid_slide:]]

    # ── 파트별 맵핑 수행
    all_assignments: dict[str, int | None] = {}  # chunk_key -> slide_no
    all_chunks: list[str] = []
    chunk_to_slide: list[int | None] = []

    for part_idx, (script_part, slide_nums) in enumerate(zip(script_parts, slide_splits)):
        if not slide_nums:
            continue

        # 이 파트에 해당하는 슬라이드 목록
        part_slide_list = "\n".join([
            l for l in slide_list_lines
            if any(f"[슬라이드 {n}]" in l for n in slide_nums)
        ])

        prompt = f"""당신은 의학 강의 대본과 슬라이드를 정밀 매핑하는 최고 전문가입니다.

강의: {lecture_info}
총 슬라이드 수: {total_pages}
이번 파트에서 처리할 슬라이드: {slide_nums}

[슬라이드 목록]
{part_slide_list}

[강의 대본]
{script_part}

[작업 지시]
위 대본을 읽고, 각 슬라이드에 해당하는 대본 구간을 정확히 분할하세요.

[핵심 규칙]
1. 대본의 키워드/주제어가 슬라이드 텍스트와 일치하는 부분을 찾아 배정하세요.
2. 슬라이드 번호는 반드시 오름차순이어야 합니다.
3. 교수님이 건너뛴 슬라이드는 배정하지 않아도 됩니다.
4. 한 슬라이드에 여러 문단이 배정될 수 있고, 여러 슬라이드에 걸쳐 설명하는 경우도 있습니다.
5. 표지/섹션구분 슬라이드는 목록에 없으므로 신경 쓰지 마세요.
6. 대본 내용을 절대 생략하거나 요약하지 말고, 원문 그대로 배정하세요.
7. 슬라이드와 무관한 잡담/공지/사설은 배정하지 않아도 됩니다.

반드시 아래 JSON 형식으로만 출력하세요. 대본 내용은 원문 그대로 넣으세요:
{{
  "mappings": [
    {{
      "slide_no": 5,
      "script": "해당 슬라이드의 대본 원문 전체"
    }},
    {{
      "slide_no": 7,
      "script": "해당 슬라이드의 대본 원문 전체"
    }}
  ]
}}
"""
        text, cost = await _generate(prompt, model="gemini-2.5-flash-lite")
        total_input += cost["input_tokens"]
        total_output += cost["output_tokens"]

        try:
            clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data.get("mappings", []):
                    sn = item.get("slide_no")
                    sc = item.get("script", "")
                    if sn and sc and sn in slide_nums:
                        chunk_key = f"slide_{sn}"
                        all_assignments[chunk_key] = sn
                        all_chunks.append(sc)
                        chunk_to_slide.append(sn)
        except Exception:
            pass

    # ── 슬라이드별 대본 합치기
    slide_scripts: dict[int, str] = {}
    for chunk_idx, slide_no in enumerate(chunk_to_slide):
        if slide_no is not None:
            if slide_no in slide_scripts:
                slide_scripts[slide_no] += "\n" + all_chunks[chunk_idx]
            else:
                slide_scripts[slide_no] = all_chunks[chunk_idx]

    page_scripts = []
    for i in range(1, total_pages + 1):
        if i in slide_scripts:
            page_scripts.append(slide_scripts[i])
        else:
            page_scripts.append("해당 없음")

    # ── 누락 구간 복구: 맵핑된 내용을 합쳐서 원본과 비교, 빠진 구간 감지 후 삽입
    page_scripts = _recover_missing_content(raw_script, page_scripts, slide_texts)

    return page_scripts, all_chunks, chunk_to_slide, calculate_cost(total_input, total_output)


def _recover_missing_content(raw_script: str, page_scripts: list[str], slide_texts: list[str]) -> list[str]:
    """
    맵핑 후 원본 대본에서 누락된 구간을 감지하여 가장 적절한 슬라이드에 삽입.
    문장 단위로 원본을 순회하며 정제본에 없는 구간을 찾아냄.
    """
    import difflib

    # 정제본 전체 텍스트 (슬라이드 구분 없이)
    all_mapped = " ".join([s for s in page_scripts if s != "해당 없음"])

    # 원본을 문단 단위로 분할
    paragraphs = [p.strip() for p in raw_script.split("\n\n") if p.strip()]

    missing_paragraphs: list[tuple[int, str]] = []  # (원본 위치, 내용)

    for i, para in enumerate(paragraphs):
        # 문단의 핵심 키워드 추출 (20자 이상인 문장의 앞 30자)
        key = para[:40].replace(" ", "")
        mapped_flat = all_mapped.replace(" ", "")

        # 정제본에 없으면 누락으로 판단
        if len(para) > 50 and key not in mapped_flat:
            missing_paragraphs.append((i, para))

    if not missing_paragraphs:
        return page_scripts

    # 누락된 문단을 앞뒤 문단 기준으로 가장 가까운 슬라이드에 삽입
    for para_idx, missing_para in missing_paragraphs:
        # 앞 문단이 어느 슬라이드에 배정됐는지 찾기
        best_slide = None
        for look_back in range(1, min(5, para_idx + 1)):
            prev_para = paragraphs[para_idx - look_back]
            prev_key = prev_para[:40].replace(" ", "")
            for slide_idx, script in enumerate(page_scripts):
                if script != "해당 없음" and prev_key in script.replace(" ", ""):
                    best_slide = slide_idx
                    break
            if best_slide is not None:
                break

        # 앞에서 못 찾으면 뒤 문단 기준으로
        if best_slide is None:
            for look_forward in range(1, min(5, len(paragraphs) - para_idx)):
                next_para = paragraphs[para_idx + look_forward]
                next_key = next_para[:40].replace(" ", "")
                for slide_idx, script in enumerate(page_scripts):
                    if script != "해당 없음" and next_key in script.replace(" ", ""):
                        best_slide = slide_idx
                        break
                if best_slide is not None:
                    break

        # 그래도 못 찾으면 해당없음이 아닌 가장 가까운 슬라이드에 붙임
        if best_slide is None:
            for offset in range(1, len(page_scripts)):
                for direction in [-1, 1]:
                    idx = len(page_scripts) // 2 + offset * direction
                    if 0 <= idx < len(page_scripts) and page_scripts[idx] != "해당 없음":
                        best_slide = idx
                        break
                if best_slide is not None:
                    break

        if best_slide is not None:
            page_scripts[best_slide] = page_scripts[best_slide] + "\n" + missing_para

    return page_scripts


async def review_mapping(
    page_scripts: list[str],
    slide_texts: list[str],
    chunks: list[str],
    chunk_assignments: list[int | None],
    lecture_info: str,
) -> tuple[list[str], dict]:
    """
    맵핑 검토: 내용이 비정상적으로 몰린 슬라이드 감지 후 재배정
    """
    total_pages = len(page_scripts)
    total_input = 0
    total_output = 0

    # 비정상 슬라이드 감지
    problem_slides = []
    for i, script in enumerate(page_scripts):
        if script == "해당 없음":
            continue
        slide_no = i + 1
        # 3000자 초과 or 앞뒤가 다 해당없음인데 혼자 너무 많음
        neighbors_empty = all(
            page_scripts[j] == "해당 없음"
            for j in range(max(0, i - 3), min(total_pages, i + 4))
            if j != i
        )
        if len(script) > 3000 or (neighbors_empty and len(script) > 500):
            problem_slides.append(slide_no)

    if not problem_slides:
        return page_scripts, calculate_cost(total_input, total_output)

    for prob_slide in problem_slides[:5]:
        script_to_review = page_scripts[prob_slide - 1]
        start = max(0, prob_slide - 5)
        end = min(total_pages, prob_slide + 4)

        nearby_slides = "\n".join([
            f"[슬라이드 {j+1}] {' '.join(slide_texts[j].split())[:300]}"
            for j in range(start, end)
        ])

        prompt = f"""당신은 강의 대본 맵핑을 검토하는 전문가입니다.

강의: {lecture_info}

현재 슬라이드 {prob_slide}에 아래 대본이 배정되어 있는데, 내용이 너무 많거나 주변 슬라이드 내용이 섞인 것 같습니다.

[주변 슬라이드 목록 (슬라이드 {start+1}~{end})]
{nearby_slides}

[현재 슬라이드 {prob_slide}에 배정된 대본]
{script_to_review}

[작업]
위 대본을 주변 슬라이드 내용과 비교하여, 각 문단/문장이 실제로 어느 슬라이드에 해당하는지 재배정하세요.
슬라이드 번호는 {start+1}~{end} 범위 내에서만, 반드시 오름차순으로 배정하세요.

반드시 아래 JSON 형식으로만 출력하세요:
{{
  "reassignments": [
    {{"slide_no": {prob_slide}, "script": "이 슬라이드에 해당하는 대본 원문"}},
    {{"slide_no": {min(prob_slide+1, total_pages)}, "script": "다음 슬라이드에 해당하는 대본 원문"}}
  ]
}}
"""
        text, cost = await _generate(prompt, model="gemini-2.5-flash-lite")
        total_input += cost["input_tokens"]
        total_output += cost["output_tokens"]

        try:
            clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data.get("reassignments", []):
                    sn = item.get("slide_no")
                    sc = item.get("script", "")
                    if sn and sc and start < sn <= end:
                        page_scripts[sn - 1] = sc
        except Exception:
            pass

    return page_scripts, calculate_cost(total_input, total_output)


async def refine_page_scripts(page_scripts: list[str], lecture_info: str) -> tuple[list[str], dict]:
    BATCH_SIZE = 8
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
아래는 {lecture_info} 강의의 슬라이드별 대본으로, 교수님 발화를 음성인식한 텍스트입니다.
각 슬라이드 대본을 아래 조건에 따라 정제하되, [슬라이드 N] 형식은 반드시 유지하세요.

[정제 조건]
1. 음성인식으로 잘못 표기된 의학 용어를 문맥을 보고 판단하여 정확한 영문으로 교정
   - 교정 형식: 영문(한국어) — 예: autoimmune hepatitis(자가면역성 간염), parietal cell(벽세포)
   - 문맥상 명확한 용어는 예외 없이 교정, 불확실하면 [?원문] 표시
2. 콩글리시나 잘못된 영어 표현을 올바른 의학 영어로 수정
3. 교수님 질문-학생 답변 대화 형식 유지
4. 불필요한 미사여구 제거, 구어체는 반드시 유지
5. 교수님 농담은 살리기
6. 교수님의 비유, 개인 경험담, 사례 얘기는 의학 내용 이해를 돕는 설명이므로 반드시 살릴 것
   제거해도 되는 것: 순수 행정공지(출석, 과제 제출), 수업 진행 안내("다음 넘어가겠습니다") 정도만
   절대 제거하면 안 되는 것: 교수님 비유/예시/경험담/농담/임상 사례 → 이건 다 살릴 것
7. 의학용어는 영어로 + 괄호 안에 한국어 번역 (예: smallpox(천연두))
8. 절대 요약하지 말 것, 산문 형식 유지
9. 강조/기억하라고 한 내용 반드시 살리기
10. "해당 없음"인 슬라이드는 그대로 "해당 없음" 출력

[슬라이드별 대본]
{combined}

정제된 슬라이드별 대본만 출력하세요. [슬라이드 N] 형식 반드시 유지하면서 본문만 출력하세요.
"""
        text, cost = await _generate(prompt, model="gemini-2.5-flash-lite")
        total_input += cost["input_tokens"]
        total_output += cost["output_tokens"]

        parts = re.split(r'\[슬라이드 \d+\]', text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == len(batch):
            all_refined.extend(parts)
        else:
            all_refined.extend(batch)

    return all_refined, calculate_cost(total_input, total_output)



async def preprocess_script(raw_script: str, lecture_info: str, slide_texts: list[str]) -> tuple[str, dict]:
    """
    맵핑 전 사전처리:
    슬라이드 텍스트를 참고해서 음성인식 대본의 의학용어를 정확하게 교정
    슬라이드 컨텍스트를 같이 주기 때문에 "포임 → POEM(경구내시경 근절개술)" 처럼
    강의 주제에 맞게 정확한 용어로 교정 가능
    """
    # 슬라이드에서 의학용어 힌트 추출 (영문 키워드 위주)
    slide_keywords = set()
    for text in slide_texts:
        # 영문 단어/약어 추출
        words = re.findall(r'[A-Z][A-Za-z0-9\-]{2,}|[A-Z]{2,}', text)
        slide_keywords.update(words[:5])  # 슬라이드당 최대 5개
    keyword_hint = ", ".join(sorted(slide_keywords)[:80])  # 최대 80개

    # 슬라이드 텍스트 요약 (주요 의학용어 컨텍스트용)
    slide_summary_lines = []
    for i, text in enumerate(slide_texts):
        text_match = re.search(r'\[텍스트\](.*?)(?=\[시각\]|\[유형\]|$)', text, re.DOTALL)
        extracted = text_match.group(1).strip()[:200] if text_match else " ".join(text.split())[:200]
        if extracted:
            slide_summary_lines.append(f"슬라이드 {i+1}: {extracted}")
    slide_summary = "\n".join(slide_summary_lines[:30])  # 최대 30개

    # 대본이 너무 길면 청크로 나눠서 처리
    CHUNK_SIZE = 8000
    if len(raw_script) <= CHUNK_SIZE:
        parts = [raw_script]
    else:
        lines = raw_script.split("\n")
        parts = []
        current = ""
        for line in lines:
            if len(current) + len(line) > CHUNK_SIZE and current:
                parts.append(current.strip())
                current = line
            else:
                current = (current + "\n" + line).strip()
        if current:
            parts.append(current.strip())

    total_input = 0
    total_output = 0
    corrected_parts = []

    for part in parts:
        prompt = f"""당신은 의학 전문가입니다. 아래는 {lecture_info} 강의의 음성인식 대본입니다.

[이 강의의 슬라이드 키워드 — 반드시 이 맥락에서 용어를 교정하세요]
{keyword_hint}

[슬라이드 내용 요약]
{slide_summary}

위 슬라이드 내용을 참고해서, 대본의 음성인식 오류를 교정하세요.
예를 들어 슬라이드에 "POEM (Per-Oral Endoscopic Myotomy)"가 있다면 대본의 "포임"은 poem(시)가 아니라 POEM(경구내시경 근절개술)로 교정해야 합니다.

[작업]
1. 슬라이드 컨텍스트를 반드시 참고하여, 한국어 발음으로 표기된 의학용어를 정확한 영문으로 교정
2. 교정 형식: 영문(한국어) 괄호 표기 — 예: autoimmune hepatitis(자가면역성 간염)
3. 처음 등장하는 의학용어에만 괄호 표기, 이후 반복은 영문만
4. 일반 한국어 문장은 그대로 유지, 구어체 그대로 유지
5. 내용을 절대 요약하거나 삭제하지 말 것

[원본 대본]
{part}

교정된 대본만 출력하세요. 원문의 모든 내용을 빠짐없이 유지하세요.
"""
        text, cost = await _generate(prompt, model="gemini-2.5-flash-lite")
        total_input += cost["input_tokens"]
        total_output += cost["output_tokens"]
        corrected_parts.append(text.strip())

    corrected_script = "\n\n".join(corrected_parts)
    return corrected_script, calculate_cost(total_input, total_output)


async def extract_emphasis(raw_script: str, lecture_info: str) -> tuple[str, dict]:
    prompt = f"""당신은 의학과 시험 준비를 돕는 전문가입니다.
아래는 {lecture_info} 강의 대본입니다.

시험에 출제될 가능성이 높은 의학적 내용만 엄격하게 발췌해 주세요.

[발췌 기준]
- "시험에 나온다", "외워라", "반드시 알아야 한다", "중요하다", "기억해라" 등 명시적으로 강조한 의학 내용
- 특정 수치, 메커니즘, 질환명, 용어를 반복 강조한 내용
- "이것만큼은", "핵심은", "포인트는" 등으로 콕 집어 언급한 내용
- 별표(★) 언급한 내용

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
    return await _generate(prompt, model="gemini-2.5-flash-lite")
