# 1. 가상환경 세팅

```bash
python3 -m venv .venv
source .venv/bin/activate
```

# 2. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

# 3. 벡터 DB 빌드

```bash
python build_db.py
```

# 4. 앱 실행

```bash
python main.py
```
