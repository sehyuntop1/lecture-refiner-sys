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


async def _generate(prompt: str, max_retries: int = 10) -> tuple[str, dict]:
    import random
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
                "429" in err_str or
                "overloaded" in err_str.lower() or
                "high demand" in err_str.lower()
            )
            if is_retryable and attempt < max_retries - 1:
                # exponential backoff + jitter (최대 60초)
                base_wait = min(2 ** attempt, 60)
                jitter = random.uniform(0, base_wait * 0.3)
                wait = base_wait + jitter
                await asyncio.sleep(wait)
                continue
            else:
                raise


async def _generate_with_image(img_bytes: bytes, prompt: str, max_retries: int = 10) -> str:
    import random
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
                "429" in err_str or
                "overloaded" in err_str.lower() or
                "high demand" in err_str.lower()
            )
            if is_retryable and attempt < max_retries - 1:
                base_wait = min(2 ** attempt, 60)
                jitter = random.uniform(0, base_wait * 0.3)
                wait = base_wait + jitter
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
                """이 강의 슬라이드를 분석해서 아래 두 가지를 출력하세요.

[텍스트] 슬라이드에 있는 모든 텍스트를 그대로 추출
[시각] 슬라이드의 핵심 시각적 특징을 한 줄로 요약
(예: '위장 해부도, 파란 화살표가 LES 가리킴' / 'GERD 병태생리 플로우차트, 빨간 박스 강조' / '내시경 사진 6장, Grade A-D 분류' / '텍스트만 있는 슬라이드' / '타임라인 다이어그램')

텍스트가 없는 이미지 슬라이드도 반드시 [시각] 설명을 작성하세요."""
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

    # ── 슬라이드 목록 (500자로 확대, 줄바꿈 제거)
    slide_list_lines = []
    for i, text in enumerate(slide_texts):
        one_line = " ".join(text.split())[:500]
        slide_list_lines.append(f"[슬라이드 {i+1}] {one_line}")
    full_slide_list = "\n".join(slide_list_lines)

    # ── 대본을 ~300자 청크로 분할 (슬라이드 전환점 정밀 감지)
    sentences = _split_script_into_sentences(raw_script)
    chunks: list[str] = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) > 300 and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sent
        else:
            current_chunk = (current_chunk + "\n" + sent).strip()
    if current_chunk:
        chunks.append(current_chunk.strip())

    total_chunks = len(chunks)

    # ── 각 청크를 담당 슬라이드에 배정
    chunk_assignments: list[int | None] = []
    prev_max_slide = 0
    prev_slide_text = ""  # 이전 배치 마지막 슬라이드 텍스트 (컨텍스트용)

    ASSIGN_BATCH = 4  # 배치 크기 줄여서 정밀도 향상
    for batch_start in range(0, total_chunks, ASSIGN_BATCH):
        batch_chunks = chunks[batch_start: batch_start + ASSIGN_BATCH]
        batch_numbered = "\n\n".join([
            f"[청크 {batch_start + j + 1}]\n{c}"
            for j, c in enumerate(batch_chunks)
        ])

        # 이전 배치 마지막 슬라이드 컨텍스트
        context_line = ""
        if prev_max_slide > 0 and prev_slide_text:
            context_line = f"\n[직전 슬라이드 {prev_max_slide} 내용]: {prev_slide_text[:200]}"

        prompt = f"""당신은 의학 강의 대본과 슬라이드를 정밀 매핑하는 전문가입니다.

강의: {lecture_info}
총 슬라이드 수: {total_pages}
현재까지 매핑 완료된 슬라이드: 1~{prev_max_slide}{context_line}

[전체 슬라이드 목록]
{full_slide_list}

[이번에 배정할 대본 청크들]
{batch_numbered}

[배정 규칙 - 엄격히 준수]
1. 각 청크의 키워드/주제가 어느 슬라이드 내용과 일치하는지 정밀하게 판단하세요.
2. 슬라이드 번호는 반드시 단조증가. 이전 번호({prev_max_slide})보다 작은 번호 절대 불가.
3. 슬라이드 전환 신호: "다음 슬라이드", "이번엔", 새로운 주제어 등장, 이전 슬라이드 키워드 소멸.
4. 애매하면 현재 슬라이드보다 +1 앞으로 당기는 것을 적극 고려하세요. (밀리는 것 방지)
5. 청크가 슬라이드와 전혀 무관(잡담, 공지 등)하면 null.
6. 건너뛴 슬라이드는 결과에 없어도 됨.

반드시 아래 JSON만 출력하세요:
{{
  "assignments": [
    {{"chunk": {batch_start + 1}, "slide_no": 3}},
    {{"chunk": {batch_start + 2}, "slide_no": 4}},
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
                        # 이전 슬라이드 텍스트 업데이트 (다음 배치 컨텍스트용)
                        prev_slide_text = " ".join(slide_texts[sn - 1].split())[:200]
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

    return page_scripts, chunks, chunk_assignments, calculate_cost(total_input, total_output)


async def review_mapping(
    page_scripts: list[str],
    slide_texts: list[str],
    chunks: list[str],
    chunk_assignments: list[int | None],
    lecture_info: str,
) -> tuple[list[str], dict]:
    """
    맵핑 검토: 이상한 슬라이드(내용이 너무 몰리거나, 앞뒤 슬라이드와 맞지 않는 경우)를 감지하고 재배정
    """
    total_pages = len(page_scripts)
    total_input = 0
    total_output = 0

    # 문제 슬라이드 감지: 내용이 너무 많거나(>2000자), 앞뒤가 해당없음인데 혼자 내용이 많은 경우
    problem_slides = []
    for i, script in enumerate(page_scripts):
        if script == "해당 없음":
            continue
        slide_no = i + 1
        # 앞뒤 5개 슬라이드가 모두 해당없음인데 혼자 내용이 비정상적으로 많음
        neighbors_empty = all(
            page_scripts[j] == "해당 없음"
            for j in range(max(0, i-3), min(total_pages, i+4))
            if j != i
        )
        if len(script) > 1500 or neighbors_empty:
            problem_slides.append(slide_no)

    if not problem_slides:
        return page_scripts, calculate_cost(total_input, total_output)

    # 문제 슬라이드 주변 청크들을 재검토
    # 해당 슬라이드에 배정된 청크 인덱스 찾기
    for prob_slide in problem_slides[:5]:  # 최대 5개만 처리
        assigned_chunks = [
            (idx, chunks[idx]) for idx, sn in enumerate(chunk_assignments)
            if sn == prob_slide
        ]
        if not assigned_chunks:
            continue

        # 주변 슬라이드 텍스트
        start = max(0, prob_slide - 4)
        end = min(total_pages, prob_slide + 3)
        nearby_slides = "
".join([
            f"[슬라이드 {j+1}] {' '.join(slide_texts[j].split())[:300]}"
            for j in range(start, end)
        ])

        chunks_text = "

".join([
            f"[청크 {idx+1}]
{c}" for idx, c in assigned_chunks
        ])

        prompt = f"""당신은 의학 강의 대본과 슬라이드 매핑을 검토하는 전문가입니다.

강의: {lecture_info}
총 슬라이드 수: {total_pages}

현재 슬라이드 {prob_slide}에 아래 청크들이 모두 배정되어 있는데, 일부는 잘못 배정된 것 같습니다.

[슬라이드 {prob_slide} 주변 슬라이드 목록]
{nearby_slides}

[슬라이드 {prob_slide}에 현재 배정된 청크들]
{chunks_text}

[검토 규칙]
1. 각 청크가 실제로 어느 슬라이드에 해당하는지 재판단하세요.
2. 슬라이드 번호는 단조증가 유지 (슬라이드 {max(1, prob_slide-3)} ~ {min(total_pages, prob_slide+2)} 범위 내)
3. 슬라이드와 전혀 무관하면 null

반드시 아래 JSON만 출력하세요:
{{
  "reassignments": [
    {{"chunk": 청크번호, "slide_no": 슬라이드번호}},
    ...
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
                for item in data.get("reassignments", []):
                    chunk_idx = int(item["chunk"]) - 1
                    new_sn = item.get("slide_no")
                    if 0 <= chunk_idx < len(chunk_assignments):
                        if new_sn is not None:
                            new_sn = max(1, min(int(new_sn), total_pages))
                        chunk_assignments[chunk_idx] = new_sn
        except Exception:
            pass

    # 재배정 후 슬라이드별로 다시 합치기
    slide_chunks: dict[int, list[str]] = {i: [] for i in range(1, total_pages + 1)}
    for idx, slide_no in enumerate(chunk_assignments):
        if slide_no is not None and 1 <= slide_no <= total_pages:
            slide_chunks[slide_no].append(chunks[idx])

    reviewed_scripts = []
    for i in range(1, total_pages + 1):
        if slide_chunks[i]:
            reviewed_scripts.append("
".join(slide_chunks[i]))
        else:
            reviewed_scripts.append("해당 없음")

    return reviewed_scripts, calculate_cost(total_input, total_output)


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

        prompt = f"""당신은 의학 전문가이자 의학과 강의 대본 정제 전문가입니다.
아래는 {lecture_info} 강의의 슬라이드별 대본으로, 음성인식으로 생성된 텍스트입니다.

[가장 중요: 음성인식 의학용어 교정]
이 대본은 교수님 발화를 음성인식한 것이므로, 의학 용어가 한국어 발음으로 잘못 표기되어 있습니다.
당신은 의학 전문 지식을 바탕으로 문맥을 보고 어떤 의학 용어인지 스스로 판단하여 교정하세요.
- 모든 의학 용어(질환명, 약물명, 해부학 용어, 시술명, 검사명 등)를 정확한 영문으로 교정
- 교정 형식: 영문(한국어) — 예: lymphoma(림프종), biopsy(생검), parietal cell(벽세포)
- 문맥상 어떤 용어인지 명확히 판단되면 예외 없이 교정
- 판단이 불가능한 경우만 [?원문] 표시

[정제 조건]
1. 위 음성인식 교정을 최우선으로 적용
2. 구어체 반드시 유지, 절대 요약하지 말 것
3. 교수님 질문-학생 답변 대화 형식 유지
4. 교수님 농담 살리기
5. 슬라이드와 무관한 잡담/공지/사설 제거 (농담 제외)
6. 강조/시험 출제 언급 반드시 살리기
7. "해당 없음"인 슬라이드는 그대로 "해당 없음" 출력

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
