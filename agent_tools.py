import subprocess
from config import search_model, HUGGINGFACE_API_KEY  # 필요에 따라 불러오기
# agent_tools.py

from utils import MEMBER_NAME_MAP, format_hhmmss

def filter_segments_by_member(segments, member_query):
    # member_query에 포함된 한국어 이름과 해당 표준 영어 이름 리스트를 생성
    standard_names = []
    for korean_name, eng_names in MEMBER_NAME_MAP.items():
        if korean_name in member_query:
            standard_names.extend(eng_names)
    filtered = []
    for seg in segments:
        faces = seg.get("faces", [])
        for face in faces:
            if "member" in face and face["member"] in standard_names:
                filtered.append(seg)
                break
    return filtered

def summarize_video_range(start_sec: float, end_sec: float, segments_meta, member_query=None):
    """
    주어진 시간 범위에 해당하는 세그먼트를 필터링하고, (옵션) 특정 인물(member_query) 관련 세그먼트만 선별하여 요약 프롬프트를 구성.
    """
    # 지정된 시간 범위에 속하는 세그먼트 필터링
    filtered = [seg for seg in segments_meta if seg["start_time"] >= start_sec and seg["end_time"] <= end_sec]
    if member_query:
        # 특정 인물 필터링 추가
        filtered = filter_segments_by_member(filtered, member_query)
    if not filtered:
        return "해당 구간에 검색 결과가 없습니다."
    
    context_lines = []
    for seg in filtered:
        start_hms = format_hhmmss(seg["start_time"])
        end_hms = format_hhmmss(seg["end_time"])
        context_lines.append(f"[세그먼트ID={seg['id']} (start_sec={seg['start_time']}, end_sec={seg['end_time']})]\n"
                             f"{start_hms} ~ {end_hms}\n{seg['caption']}")
    context = "\n".join(context_lines)
    
    prompt = f"""
Context:
{context}

요청:
위 구간의 캡션 정보를 기반으로, {member_query if member_query else "전체"}의 주요 장면과 등장 인물 정보를 간단하게 요약해 주세요.
    """
    # 실제 요약 처리: 여기서 LLM 호출 등을 추가하면 됩니다.
    # 예: response = llm.invoke(prompt)
    # return response.strip()
    return prompt  # 테스트용으로 프롬프트를 반환


def extract_video_clip(video_path: str, start_sec: float, end_sec: float, output_path: str):
    # 만약 ffmpeg가 시스템 PATH에 없다면, 전체 경로를 지정합니다.
    ffmpeg_executable = "ffmpeg"  # 또는 "C:\\ffmpeg\\bin\\ffmpeg.exe"
    command = [
        ffmpeg_executable,
        "-y",  # 덮어쓰기 허용
        "-i", video_path,
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-c", "copy",
        output_path
    ]
    try:
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        return f"영상 클립 생성 실패: {e}"

