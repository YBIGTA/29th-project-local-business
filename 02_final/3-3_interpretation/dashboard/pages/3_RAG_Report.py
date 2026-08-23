import os
import re
import sys

import streamlit as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_PIPELINE_DIR = os.path.normpath(
    os.path.join(THIS_DIR, "..", "..", "..", "3-2_model_improvement", "rag_pipeline")
)
sys.path.insert(0, RAG_PIPELINE_DIR)

from compare_products import answer_compare_query
from generate_report import generate_report
from risky_products_query import answer_risky_products_query
from topic_query import answer_topic_query
from trend_query import answer_trend_query

st.set_page_config(page_title="AI 리포트", layout="wide")

# ================= 커스텀 스타일 =================
st.markdown("""
<style>
h1 { font-size: 30px !important; font-weight: 700 !important; }
[data-testid="stCaptionContainer"] { font-size: 20px; font-weight: 500; }
.caption-small { font-size: 15px; font-weight: 500; }

.report-section-bar {
    height: 4px;
    border-radius: 2px;
    margin: -1px -1px 12px -1px;
}
.report-section-title {
    font-weight: 700;
    font-size: 16px;
    margin: 0 0 0px 0 !important;
}
.report-section-divider {
    border: none;
    border-top: 1px solid #E5E5E5;
    margin: 0 10px 16px 10px !important;
}
.report-section-body { font-size: 14px; line-height: 1.7; }
.report-section-body blockquote {
    border-left: 3px solid #ddd;
    margin: 0 0 0 0 !important;
    padding-left: 12px;
    color: #555;
    font-size: 13px;
}
.report-section-body blockquote p { margin: 0 !important; }

[class*="st-key-report-card"] {
    padding: 20px 24px !important;
}

.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.logo("assets/amazon_logo.png", size="large")

st.title("위험 리포트")
st.caption("Appliances")

SECTION_BAR_COLOR = "#232F3E"  # 모든 리포트 섹션에 공통으로 쓰는 중립 톤 (색은 위험도 배지에만 의미있게 사용)


def render_report_card(report_text: str):
    """[위험 요약] [핵심 원인] [근거 리뷰] [우선 조치 제안] 고정 형식을 하나의 박스 안에 순서대로 렌더링."""
    pattern = r"\[(.*?)\]\s*\n?(.*?)(?=\n\[|\Z)"
    sections = re.findall(pattern, report_text.strip(), re.S)

    if not sections:
        # 형식이 안 맞으면(예: 형식 밖 텍스트) 원본 그대로 표시
        st.markdown(report_text)
        return

    with st.container(border=True, key="report-card"):
        st.markdown(
            f'<div class="report-section-bar" style="background-color:{SECTION_BAR_COLOR};"></div>',
            unsafe_allow_html=True,
        )
        for i, (title, body) in enumerate(sections):
            title = title.strip()
            st.markdown(
                f'<p class="report-section-title" style="color:{SECTION_BAR_COLOR};">{title}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(body.strip())
            if i < len(sections) - 1:
                st.markdown('<hr class="report-section-divider">', unsafe_allow_html=True)


def render_answer_card(text: str, key: str):
    """단일 문단 답변(토픽/위험목록/추이/비교)을 카드 하나로 감싸서 렌더링."""
    with st.container(border=True, key=key):
        st.markdown(
            f'<div class="report-section-bar" style="background-color:{SECTION_BAR_COLOR};"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(text)


VALID_TOPICS = [
    "durability_failure", "build_quality", "incompatibility", "performance",
    "leak_damage", "water_taste_filter", "shipping_packaging",
    "return_intent", "return_blocked", "refund_warranty_cs",
    "misrepresentation", "purchase_warning",
]

report_type = st.selectbox(
    "질문 유형 선택",
    options=["기본 위험 리포트", "특정 불만 토픽 리뷰", "위험 상품 목록 요약", "위험도 추이 분석", "상품 비교 (관리자용)"],
)

default_asin = st.session_state.get("selected_asin", "")

st.divider()

# ================= 1. 기본 위험 리포트 =================
if report_type == "기본 위험 리포트":
    asin = st.text_input("parent_asin", value=default_asin)
    year_month = st.text_input("연-월 (선택, 예: 2022-06)")
    user_question = st.text_area("추가로 궁금한 점 (선택)")

    if st.button("리포트 생성", type="primary"):
        if not asin:
            st.warning("parent_asin을 입력하세요.")
        else:
            with st.spinner("리포트 생성 중... (LLM 호출이라 몇 초 걸릴 수 있습니다)"):
                result = generate_report(
                    asin, year_month=year_month or None, user_question=user_question or None
                )
            st.session_state["last_report_result"] = result
            st.session_state["last_report_asin"] = asin

    if "last_report_result" in st.session_state:
        result = st.session_state["last_report_result"]
        if result.get("error"):
            st.error(f"리포트 생성 실패: {result['error']}")
        else:
            context = result.get("context", {})
            risk_prob = context.get("risk_probability")
            if risk_prob is not None:
                badge_color = "#D62728" if risk_prob >= 0.7 else "#FF9900" if risk_prob >= 0.4 else "#2CA02C"
                st.markdown(
                    f'<p style="font-size:14px; color:#888; margin-bottom:4px;">{context.get("parent_asin", st.session_state.get("last_report_asin", asin))}</p>'
                    f'<p style="font-size:28px; font-weight:800; color:{badge_color}; margin:0 0 16px 0;">위험도 {risk_prob:.1%}</p>',
                    unsafe_allow_html=True,
                )

            render_report_card(result["report"])

            with st.expander("근거 데이터 원본 (context)"):
                st.json(result["context"])

# ================= 2. 특정 불만 토픽 리뷰 =================
elif report_type == "특정 불만 토픽 리뷰":
    asin = st.text_input("parent_asin", value=default_asin)
    topic = st.selectbox("토픽 선택", options=VALID_TOPICS)
    year_month = st.text_input("연-월 (선택, 예: 2022-06)")
    show_all = st.checkbox("더 많은 리뷰 보기 (최대 15건, 기본은 5건)")

    if st.button("조회", type="primary"):
        if not asin:
            st.warning("parent_asin을 입력하세요.")
        else:
            with st.spinner("리뷰 분석 중..."):
                result = answer_topic_query(
                    asin, topic, year_month=year_month or None, show_all=show_all
                )
            st.session_state["last_topic_result"] = result

    if "last_topic_result" in st.session_state:
        result = st.session_state["last_topic_result"]
        if result.get("error"):
            st.error(result["error"])
        else:
            render_answer_card(result["answer"], key="topic-card")
            st.markdown(
                f'<p class="caption-small">조회된 리뷰 수: {result["review_count"]}건 · {result["review_count_note"]}</p>',
                unsafe_allow_html=True,
            )

# ================= 3. 위험 상품 목록 요약 =================
elif report_type == "위험 상품 목록 요약":
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "정렬 기준", options=["risk", "confidence_weighted"],
            format_func=lambda x: "위험도순" if x == "risk" else "위험도×신뢰도순",
        )
    with col2:
        limit = st.slider("조회할 상품 수", min_value=3, max_value=20, value=5)
    year_month = st.text_input("연-월 (선택, 비워두면 최신 월)")

    if st.button("요약 생성", type="primary"):
        with st.spinner("위험 상품 목록 분석 중..."):
            result = answer_risky_products_query(
                sort_by=sort_by, limit=limit, year_month=year_month or None
            )
        st.session_state["last_risky_result"] = result

    if "last_risky_result" in st.session_state:
        result = st.session_state["last_risky_result"]
        if result.get("error"):
            st.error(result["error"])
        else:
            st.markdown(
                f'<p class="caption-small">기준 월: {result.get("year_month")} · 조회된 상품 수: {result["product_count"]}건</p>',
                unsafe_allow_html=True,
            )
            render_answer_card(result["answer"], key="risky-card")

# ================= 4. 위험도 추이 분석 =================
elif report_type == "위험도 추이 분석":
    asin = st.text_input("parent_asin", value=default_asin)

    if st.button("추이 분석", type="primary"):
        if not asin:
            st.warning("parent_asin을 입력하세요.")
        else:
            with st.spinner("추이 분석 중..."):
                result = answer_trend_query(asin)
            st.session_state["last_trend_result"] = result

    if "last_trend_result" in st.session_state:
        result = st.session_state["last_trend_result"]
        if result.get("error"):
            st.error(result["error"])
        else:
            render_answer_card(result["answer"], key="trend-card")
            with st.expander("분석 상세 데이터 (analysis)"):
                st.json(result["analysis"])

# ================= 5. 상품 비교 (관리자용) =================
else:
    col1, col2 = st.columns(2)
    with col1:
        asin_a = st.text_input("상품 A parent_asin")
        year_month_a = st.text_input("상품 A 연-월 (선택)")
    with col2:
        asin_b = st.text_input("상품 B parent_asin")
        year_month_b = st.text_input("상품 B 연-월 (선택)")

    if st.button("비교하기", type="primary"):
        if not asin_a or not asin_b:
            st.warning("두 상품의 parent_asin을 모두 입력하세요.")
        else:
            with st.spinner("두 상품 비교 분석 중..."):
                result = answer_compare_query(
                    asin_a, asin_b,
                    year_month_a=year_month_a or None, year_month_b=year_month_b or None,
                )
            st.session_state["last_compare_result"] = result

    if "last_compare_result" in st.session_state:
        result = st.session_state["last_compare_result"]
        if result.get("error"):
            st.error(result["error"])
        else:
            render_answer_card(result["answer"], key="compare-card")
            with st.expander("비교 상세 데이터 (comparison)"):
                st.json(result["comparison"])