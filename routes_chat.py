from flask import Blueprint, request, jsonify
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# [변경] 최신 모듈로부터 임포트 (langchain-community 및 langchain-huggingface)
from langchain_community.embeddings import HuggingFaceEmbeddings  # (옵션)
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory

from config import search_model, HUGGINGFACE_API_KEY  # config에서 API 키 불러오기
from utils import get_db_connection, format_hhmmss, MEMBER_NAME_MAP

chat_bp = Blueprint('chat_bp', __name__)

#########################################
# [변경] 전역 FAISS 인덱스 및 메타데이터 구축
#########################################

# 1. SBERT 모델 로드 (임베딩 계산용)
embed_model = SentenceTransformer("bongsoo/kpf-sbert-128d-v1")

# 2. 세그먼트 메타데이터(JSON 파일) 로드
with open("data/combined_metadata.json", "r", encoding="utf-8") as f:
    combined_data = json.load(f)

# 3. 세그먼트 정보와 임베딩 벡터 리스트 생성
segments_meta = []       # 각 세그먼트의 추가 정보 저장 (video_id, start_time, caption, 등)
embeddings_list = []     # 각 캡션의 128차원 임베딩

for seg in combined_data:
    video_id = seg.get("video_id", "default_video")
    timestamp = float(seg.get("timestamp", 0))
    caption = seg.get("caption", "")
    seg_info = {
        "video_id": video_id,
        "start_time": timestamp,
        "end_time": timestamp + 1.0,  # 여기서는 1초 길이라고 가정
        "caption": caption,
        "faces": seg.get("faces", [])
    }
    segments_meta.append(seg_info)
    emb = embed_model.encode(caption)
    embeddings_list.append(emb)

embeddings_array = np.vstack(embeddings_list).astype('float32')
d = embeddings_array.shape[1]  # 128차원
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(embeddings_array)

#########################################
# [변경] LangChain 기반 LLM 구성 및 검색 함수
#########################################
# 최신 HuggingFaceEndpoint를 사용하여 LLM 구성
# max_new_tokens를 명시적으로 150으로 지정하여 오류를 피합니다.
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.7,
    max_new_tokens=150,
    task="text2text-generation",  # 작업을 명시적으로 지정
    model_kwargs={}               # 불필요한 파라미터를 보내지 않음
)



# (선택사항) 대화 메모리 추가 – 멀티턴 대화 지원
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def faiss_retriever(query, top_k=4):
    query_vec = embed_model.encode(query).astype('float32')
    query_vec = np.expand_dims(query_vec, axis=0)
    distances, indices = faiss_index.search(query_vec, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(segments_meta):
            results.append(segments_meta[idx])
    return results

def generate_answer_with_retrieval(query: str) -> str:
    retrieved_segments = faiss_retriever(query, top_k=4)
    context_lines = []
    for seg in retrieved_segments:
        start_hms = format_hhmmss(seg["start_time"])
        end_hms = format_hhmmss(seg["end_time"])
        cap = seg["caption"]
        context_lines.append(f"[{seg['video_id']} {start_hms} ~ {end_hms}] {cap}")
    context_text = "\n".join(context_lines)
    
    # 새 프롬프트 구성: 검색된 문맥(context)와 사용자 입력 결합
    prompt = f"""
Context:
{context_text}

사용자 입력: "{query}"

요청:
위 문맥을 바탕으로, 검색어와 가장 유사한 장면에 대한 내용을 자연스러운 한국어로 요약해 주세요.
    """
    response = llm.invoke(prompt)
    return response.strip()

#########################################
# 기존 helper 함수 (변경 없음)
#########################################
def member_matches_query(members, query):
    query_lower = query.lower()
    for member in members:
        if member.lower() in query_lower or query_lower in member.lower():
            return True
        for korean_name, standard_names in MEMBER_NAME_MAP.items():
            if korean_name in query_lower and member in standard_names:
                return True
    return False

#########################################
# Flask 라우트: /chat
#########################################
@chat_bp.route("/chat", methods=["POST"])
def unified_chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    # 1) 수정 명령 ("수정:" 모드) - 기존 수정 로직 그대로 유지
    override_pattern = r'^수정:\s*세그먼트ID=(\d+)\s*내용=(.*?)(?:\s+멤버=(.*))?$'
    if user_msg.startswith("수정:"):
        match = re.match(override_pattern, user_msg)
        if match:
            seg_id_str = match.group(1)
            override_text = match.group(2).strip()
            members_text = match.group(3)
            try:
                seg_id = int(seg_id_str)
            except ValueError:
                seg_id = None
            if not seg_id or not override_text:
                return jsonify({"response": "교정 명령 구문이 잘못되었습니다."})
            conn = get_db_connection()
            cur = conn.cursor()
            select_sql = """
            SELECT caption, manual_caption, faces
            FROM njz_segments
            WHERE id = %s
            """
            cur.execute(select_sql, (seg_id,))
            row = cur.fetchone()
            if not row:
                cur.close()
                conn.close()
                return jsonify({"response": f"세그먼트 {seg_id}를 찾을 수 없습니다."})
            old_auto_cap = row[0] or ""
            old_manual_cap = row[1] or ""
            old_faces = row[2] or "[]"
            old_final_cap = old_manual_cap.strip() if old_manual_cap.strip() else old_auto_cap
            if isinstance(old_faces, str):
                try:
                    old_faces = json.loads(old_faces)
                except:
                    old_faces = []
            if members_text:
                members_list = [m.strip() for m in members_text.split(",") if m.strip()]
                new_faces = json.dumps([{"member": m} for m in members_list])
            else:
                new_faces = json.dumps([])
            new_emb = search_model.encode(override_text)
            new_emb_str = str(new_emb.tolist())
            if members_text:
                update_sql = """
                UPDATE njz_segments
                SET manual_caption = %s,
                    embedding = %s,
                    faces = %s
                WHERE id = %s
                """
                cur.execute(update_sql, (override_text, new_emb_str, new_faces, seg_id))
            else:
                update_sql = """
                UPDATE njz_segments
                SET manual_caption = %s,
                    embedding = %s
                WHERE id = %s
                """
                cur.execute(update_sql, (override_text, new_emb_str, seg_id))
            hist_sql = """
            INSERT INTO njz_segment_operations_history
            (operation_type, segment_ids_before, segment_ids_after, old_captions, new_captions, old_faces, new_faces, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cur.execute(hist_sql, (
                "update",
                json.dumps([seg_id]),
                json.dumps([seg_id]),
                json.dumps([old_final_cap]),
                json.dumps([override_text]),
                json.dumps([old_faces]),
                json.dumps([json.loads(new_faces)]),
                "local_user"
            ))
            conn.commit()
            cur.close()
            conn.close()
            return jsonify({"response": f"세그먼트 {seg_id} 캡션이 수정되었습니다."})
        else:
            return jsonify({"response": "수정 명령 형식이 올바르지 않습니다."})

    # 2) 질문 ("질문:" 모드) - 기존 질문 로직 그대로 유지
    elif user_msg.startswith("질문:"):
        question = user_msg[len("질문:"):].strip()
        conn = get_db_connection()
        cur = conn.cursor()
        sql = """
        SELECT id, start_time, end_time, caption, manual_caption, faces
        FROM njz_segments
        ORDER BY start_time ASC
        """
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        grouped_segments = []
        current_group = []
        THRESHOLD = 1.0
        for row in rows:
            seg_id = row[0]
            start_time = row[1]
            end_time = row[2]
            auto_cap = row[3] or ""
            manual_cap = row[4] or ""
            faces_json = row[5] or "[]"
            final_cap = manual_cap.strip() if manual_cap.strip() else auto_cap
            if isinstance(faces_json, str):
                try:
                    faces_data = json.loads(faces_json)
                except:
                    faces_data = []
            else:
                faces_data = faces_json
            members = [f["member"] for f in faces_data if "member" in f]
            seg_info = {
                "id": seg_id,
                "start": start_time,
                "end": end_time,
                "cap": final_cap,
                "members": members,
            }
            if not current_group:
                current_group.append(seg_info)
            else:
                last = current_group[-1]
                time_gap = seg_info["start"] - last["end"]
                if time_gap <= THRESHOLD:
                    current_group[-1]["cap"] = f"{current_group[-1]['cap']} / {seg_info['cap']}"
                    current_group[-1]["members"] = list(set(current_group[-1]["members"] + seg_info["members"]))
                    current_group[-1]["end"] = seg_info["end"]
                    current_group[-1]["ids"] = current_group[-1].get("ids", [last["id"]]) + [seg_info["id"]]
                else:
                    grouped_segments.append(current_group)
                    current_group = [seg_info]
        if current_group:
            grouped_segments.append(current_group)
        target_keys = [k for k in MEMBER_NAME_MAP.keys() if k in question]
        if target_keys:
            target_names = set()
            for k in target_keys:
                target_names.update(MEMBER_NAME_MAP[k])
            filtered_rows = []
            for row in rows:
                faces_json = row[5] or "[]"
                try:
                    faces_data = json.loads(faces_json) if isinstance(faces_json, str) else faces_json
                except:
                    faces_data = []
                members = [f["member"] for f in faces_data if "member" in f]
                if set(members) & target_names:
                    filtered_rows.append(row)
            if filtered_rows:
                rows = filtered_rows
            else:
                return jsonify({"response": "해당 멤버가 등장하는 세그먼트가 없습니다."})
        summary_lines = []
        for idx, group in enumerate(grouped_segments, start=1):
            if "ids" in group[0]:
                first_id = group[0]["ids"][0]
                last_id = group[0]["ids"][-1]
            else:
                first_id = group[0]["id"]
                last_id = group[0]["id"]
            start_hms = format_hhmmss(group[0]["start"])
            end_hms = format_hhmmss(group[-1]["end"])
            members_set = set()
            for seg in group:
                members_set.update(seg["members"])
            member_str = ", ".join(sorted(members_set)) if members_set else "없음"
            cap_summary = group[0]["cap"]
            summary_lines.append(f"그룹 {idx}: 세그먼트ID={first_id}~{last_id} ({start_hms} ~ {end_hms}), 캡션: {cap_summary}, 등장인물: {member_str}")
        summary_text = "\n".join(summary_lines)
        response_text = f"검색 결과:\n{summary_text}"
        return jsonify({"response": response_text})

    # 3) 기본 검색 로직 - [변경] FAISS 및 LangChain 기반 RAG 방식 적용
    else:
        chat_response = generate_answer_with_retrieval(user_msg)
        return jsonify({"response": chat_response})


@chat_bp.route("/segment/group_modify", methods=["POST"])
def group_modify():
    data = request.get_json()
    segment_ids = data.get("segment_ids")
    new_caption = data.get("new_caption", "").strip()
    new_members = data.get("new_members", [])
    if not segment_ids or not new_caption:
        return jsonify({"success": False, "message": "세그먼트 ID와 새 캡션이 필요합니다."}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    old_caps = []
    old_faces_list = []
    for seg_id in segment_ids:
        cur.execute("SELECT caption, manual_caption, faces FROM njz_segments WHERE id = %s", (seg_id,))
        row = cur.fetchone()
        if not row:
            continue
        old_auto_cap = row[0] or ""
        old_manual_cap = row[1] or ""
        old_faces = row[2] or "[]"
        old_final_cap = old_manual_cap.strip() if old_manual_cap.strip() else old_auto_cap
        if isinstance(old_faces, str):
            try:
                old_faces = json.loads(old_faces)
            except:
                old_faces = []
        old_caps.append(old_final_cap)
        old_faces_list.append(old_faces)
        new_emb = search_model.encode(new_caption)
        new_emb_str = str(new_emb.tolist())
        if new_members:
            new_faces = json.dumps([{"member": m} for m in new_members])
            cur.execute("UPDATE njz_segments SET manual_caption=%s, embedding=%s, faces=%s WHERE id = %s",
                        (new_caption, new_emb_str, new_faces, seg_id))
        else:
            cur.execute("UPDATE njz_segments SET manual_caption=%s, embedding=%s WHERE id = %s",
                        (new_caption, new_emb_str, seg_id))
    hist_sql = """
    INSERT INTO njz_segment_operations_history 
    (operation_type, segment_ids_before, segment_ids_after, old_captions, new_captions, old_faces, new_faces, created_by)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    new_faces_val = json.dumps([{"member": m} for m in new_members]) if new_members else json.dumps([])
    cur.execute(hist_sql, (
        "group_update",
        json.dumps(segment_ids),
        json.dumps(segment_ids),
        json.dumps(old_caps),
        json.dumps([new_caption] * len(segment_ids)),
        json.dumps(old_faces_list),
        json.dumps(json.loads(new_faces_val)),
        "local_user"
    ))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"success": True, "message": "그룹 수정이 완료되었습니다."})
