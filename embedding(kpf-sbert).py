import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. 모델 로드 (Hugging Face SBERT 모델)
model = SentenceTransformer("bongsoo/kpf-sbert-128d-v1")

# 2. JSON 데이터 불러오기 (combined metadata)
json_path = "data/combined_metadata.json"
with open(json_path, "r", encoding="utf-8") as f:
    combined_data = json.load(f)

# 3. 세그먼트 메타데이터와 임베딩 리스트 생성
segments_meta = []
embeddings_list = []

for seg in combined_data:
    video_id = seg.get("video_id", "default_video")
    timestamp = float(seg.get("timestamp", 0))
    caption = seg.get("caption", "")
    seg_info = {
        "video_id": video_id,
        "start_time": timestamp,
        "end_time": timestamp + 1.0,
        "caption": caption,
        "faces": seg.get("faces", [])
    }
    segments_meta.append(seg_info)
    # 캡션 임베딩 계산 (128차원)
    embedding = model.encode(caption)
    embeddings_list.append(embedding)

embeddings_array = np.vstack(embeddings_list).astype('float32')

# 4. FAISS 인덱스 구축 (L2 거리 기준)
d = embeddings_array.shape[1]  # 128
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(embeddings_array)

# (선택) 인덱스와 메타데이터를 파일로 저장해 둘 수 있음 (피클 저장 등)
import pickle
with open("faiss_index.pkl", "wb") as f:
    pickle.dump({"index": faiss_index, "meta": segments_meta}, f)

print("✅ FAISS 인덱스 생성 및 메타데이터 준비 완료.")
