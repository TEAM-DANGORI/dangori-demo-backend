import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer  # 구글 라이브러리 대신 추가


def build_and_save_db():
    """CSV로부터 지식 베이스를 구축하고 디스크에 저장합니다. (로컬 모델 사용)"""
    print("지식 베이스 구축 및 저장을 시작합니다... (로컬 임베딩 모델 사용)")

    # 1. CSV 파일 로드
    try:
        df = pd.read_csv("chat_history.csv")
    except FileNotFoundError:
        print(
            "오류: 'chat_history.csv' 파일을 찾을 수 없습니다. 파일이 현재 폴더에 있는지 확인해주세요."
        )
        return

    documents = (
        "카테고리: "
        + df["category"]
        + ", 고객 질문: "
        + df["query"]
        + ", 매장 답변: "
        + df["response"]
    ).tolist()

    # 2. 로컬 임베딩 모델 로드
    # 모델을 처음 실행할 때 인터넷에서 자동으로 다운로드하며, 몇 분 정도 소요될 수 있습니다.
    # 그 이후에는 저장된 모델을 바로 불러옵니다.
    print("임베딩 모델을 로딩합니다... (최초 실행 시 시간이 걸릴 수 있습니다)")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # 3. 문서 임베딩 (Google API 호출 없음)
    print("문서 임베딩을 시작합니다...")
    embeddings = model.encode(documents, show_progress_bar=True)
    print("임베딩 완료!")

    # 4. ChromaDB에 저장
    # './chroma_db' 라는 폴더에 데이터가 저장됩니다.
    client = chromadb.PersistentClient(path="./chroma_db")

    # 기존 컬렉션이 있다면 삭제 후 새로 생성
    try:
        client.delete_collection("salon_qa")
        print("기존 컬렉션을 삭제했습니다.")
    except Exception:
        print("기존 컬렉션이 없어 새로 생성합니다.")

    collection = client.create_collection("salon_qa")
    collection.add(
        ids=[f"doc_{i}" for i in range(len(documents))],
        embeddings=embeddings.tolist(),  # NumPy 배열을 리스트로 변환
        documents=documents,
    )
    print("지식 베이스 구축 및 디스크 저장 완료! (./chroma_db)")


if __name__ == "__main__":
    build_and_save_db()
