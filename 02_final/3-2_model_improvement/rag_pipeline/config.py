"""
3-2 공통 설정 파일

.env 파일에서 DB 접속 정보와 OpenAI API 키를 읽어온다.
retrieval.py, generate_report.py 등에서 이 파일을 import해서 사용한다.

실행 전 준비:
    pip install python-dotenv sqlalchemy pymysql openai
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# .env 파일 읽어오기 (이 파일과 같은 폴더에 .env가 있어야 함)
load_dotenv()

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- MySQL ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "amazon_risk_service")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:3306/{DB_NAME}?charset=utf8mb4"
)

# 값이 잘 읽혔는지 확인용 (실행 시 한 번 체크)
if __name__ == "__main__":
    print("OPENAI_API_KEY 로드됨:", bool(OPENAI_API_KEY))
    print("DB_HOST:", DB_HOST)
    print("DB_USER:", DB_USER)
    print("DB_NAME:", DB_NAME)
