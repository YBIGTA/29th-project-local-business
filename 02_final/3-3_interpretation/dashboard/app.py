import streamlit as st
from utils.api_client import get_health, API_BASE_URL

st.set_page_config(
    page_title="Appliances 품질 위험 대시보드",
    page_icon="⚠️",
    layout="wide",
)

st.title("⚠️ Amazon Appliances 품질 위험 대시보드")

st.markdown(
    """
    이 앱은 **FastAPI 백엔드(3-1)** 가 제공하는 API만 호출해서 화면을 그립니다.
    왼쪽 사이드바에서 페이지를 이동하세요.

    - **전체 현황**: 고위험 상품 목록, 위험도 순위
    - **상품 상세**: 특정 상품의 위험도 추이, 불만 토픽, 근거 리뷰, 위험 판단 근거
    """
)

st.divider()

st.subheader("백엔드 연결 상태")
health = get_health()
if health:
    st.success(f"API 서버 정상 연결됨 ({API_BASE_URL})")
    st.json(health)
else:
    st.error(
        f"API 서버({API_BASE_URL})에 연결할 수 없습니다. "
        "터미널에서 `uvicorn main:app --reload`로 FastAPI가 켜져 있는지 확인하세요."
    )