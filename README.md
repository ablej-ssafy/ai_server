# AI SERVER

## Setting

### windows(venv)

- 로컬 개발 환경
  - PyCharm 2024.01.04
  - Python 3.10.6
  - pip 22.2.1

```shell
python -m venv venv

.\venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

# 추가적인 pip install 진행 시 사용
pip freeze > requirements.txt

# 서버 실행
uvicorn app.main:app --log-level debug
|
python main.py
```

# Feature

## Git Analysis
