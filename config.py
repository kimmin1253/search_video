import json
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# config.json에서 API 키 로드
with open('config.json', 'r') as f:
    config_json = json.load(f)

GEMINI_API_KEY = config_json["GEMINI_API_KEY"]
SLACK_API_KEY = config_json["SLACK_API_KEY"]
HUGGINGFACE_API_KEY = config_json["HUGGINGFACE_API_KEY"]

# Gemini API 설정 (기존 기능)
genai.configure(api_key=GEMINI_API_KEY)

# SBERT 임베딩 모델 (검색용)
search_model = SentenceTransformer("bongsoo/kpf-sbert-128d-v1")
