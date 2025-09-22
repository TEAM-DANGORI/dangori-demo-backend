import os
import google.generativeai as genai
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer  # 추가된 라이브러리

# .env 파일에서 Gemini API 키를 불러옵니다.
load_dotenv()

# --- 1. 초기 설정 및 전역 변수 ---

# 답변 생성을 위한 Gemini API 키 설정
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "API 키가 설정되지 않았습니다. .env 파일이나 환경 변수를 확인해주세요."
    )
genai.configure(api_key=api_key)

# 전역 변수로 로컬 임베딩 모델과 DB 컬렉션을 선언
embedding_model = None
db_collection = None

# --- 2. FastAPI 앱 설정 ---

app = FastAPI(
    title="뷰티샵 RAG 챗봇 API (로컬 임베딩)",
    description="SentenceTransformer와 Gemini를 사용하여 과거 채팅 기록을 바탕으로 답변하는 챗봇입니다.",
)


# Pydantic 모델: API 요청 본문의 형식을 정의
class ChatQuery(BaseModel):
    query: str


# FastAPI의 생명주기 이벤트: 서버가 시작될 때 실행되는 함수
@app.on_event("startup")
def startup_event():
    """서버 시작 시, 로컬 임베딩 모델과 디스크에 저장된 ChromaDB를 불러옵니다."""
    global db_collection, embedding_model

    # 1. 로컬 임베딩 모델 로드 (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있음)
    print("임베딩 모델을 로딩합니다...")
    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    print("임베딩 모델 로딩 완료!")

    # 2. './chroma_db' 경로에서 저장된 벡터 DB를 불러옵니다.
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        db_collection = client.get_collection("salon_qa")
        print("저장된 벡터 DB 로딩 성공!")
    except Exception as e:
        print(f"DB 로딩 실패: {e}")
        print(
            "에러 발생! 'python build_db.py'를 먼저 실행하여 DB를 생성했는지 확인해주세요."
        )


# --- 3. RAG 답변 생성 및 API 엔드포인트 ---


def get_rag_response(collection, user_query: str, top_k: int = 3):
    """사용자 질문에 대해 RAG 기반의 답변을 생성합니다."""
    # startup 시점에 모델/DB가 로드되지 않았을 경우를 대비한 방어 코드
    if collection is None or embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail="지식 베이스 또는 임베딩 모델이 준비되지 않았습니다.",
        )

    # 1. 사용자 질문을 "로컬 모델"로 임베딩합니다.
    query_embedding = embedding_model.encode(user_query)

    # 2. 벡터 DB에서 유사도가 높은 문서를 검색합니다.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],  # NumPy 배열을 리스트로 변환
        n_results=top_k,
    )
    retrieved_docs = "\n".join(results["documents"][0])

    # 3. 검색된 문서(Context)와 사용자 질문을 합쳐 Gemini에게 전달할 프롬프트를 구성합니다.
    prompt = f"""
    당신은 친절한 미용실 AI 상담원입니다.
    아래에 제공된 '과거 상담 내역'을 바탕으로 사용자의 질문에 가장 적절하고 자연스럽게 답변해주세요.
    상담 내역에 없는 내용이라면, 아는 것처럼 꾸며내지 말고 "매장에 직접 문의해보시는 게 좋겠습니다." 라고 솔직하게 답변하세요.

    ---
    [과거 상담 내역]
    {retrieved_docs}
    ---

    [사용자 질문]
    {user_query}

    [AI 상담원 답변]
    """

    # 4. "Gemini 모델"을 호출하여 최종 답변을 생성합니다.
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text


@app.post("/chat", summary="챗봇에게 질문하기")
async def chat_endpoint(request: ChatQuery):
    """사용자의 질문을 받아 RAG 챗봇의 답변을 반환합니다."""
    try:
        answer = get_rag_response(db_collection, request.query)
        return {"answer": answer}
    except Exception as e:
        # 실제 운영 환경에서는 더 상세한 에러 로깅이 필요합니다.
        raise HTTPException(status_code=500, detail=str(e))
