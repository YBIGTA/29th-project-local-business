import pandas as pd
import streamlit as st
import plotly.express as px 

from utils.api_client import (
    get_evidence_reviews,
    get_product_summary,
    get_product_topics,
    get_product_trend,
    get_risk_explanation,
)

VALID_TOPICS = [
    "durability_failure", "build_quality", "incompatibility", "performance",
    "leak_damage", "water_taste_filter", "shipping_packaging",
    "return_intent", "return_blocked", "refund_warranty_cs",
    "misrepresentation", "purchase_warning",
]

st.set_page_config(page_title="상품 상세", layout="wide")


# ================= 커스텀 스타일 =================
st.markdown("""
<style>
h1 { font-size: 30px !important; font-weight: 700 !important; }
h3 { font-size: 20px !important; font-weight: 700 !important; }
[data-testid="stCaptionContainer"] { font-size: 20px; font-weight: 500; }

.caption-small { font-size: 15px; font-weight: 500; }
.product-name { font-size: 20px !important; font-weight: 700 !important; margin: 0; }
.product-meta { font-size: 12px !important; font-weight: 400 !important; color: #888; margin: 0; }

[data-testid="stMetricValue"] { font-size: 26px; font-weight: 600; }

/* kpi로 시작하는 모든 container key에 배경 적용 (페이지 간 재사용을 위해 접두어 매칭으로 일반화) */
[class*="st-key-kpi"] {
    background-color: #F5F5F5;
}

.risk-up-metric { padding: 2px 0; }
.risk-up-delta { font-size: 23px !important; font-weight: 600; margin: 0 !important; }
.risk-up-pill {
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    margin-top: 0px;
}
.rank-badge {
    color: #999;
    font-weight: 700;
    font-size: 17px;
    margin-right: 8px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 6px 14px !important;
}

.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.logo("assets/amazon_logo.png", size="large")

st.title("상품 상세")
st.caption("Appliances")

default_asin = st.session_state.get("selected_asin", "")
asin = st.text_input("ASIN", value=default_asin)

if not asin:
    st.info("전체 현황 페이지에서 상품을 선택하거나, ASIN을 직접 입력하세요.")
    st.stop()

st.session_state["selected_asin"] = asin


# ================= 사이드바 =================
_latest_check = get_product_summary(asin, year_month=None)
latest_ym = _latest_check.get("year_month") if _latest_check else None

years_list = [str(y) for y in range(2016, 2024)]
months_list = [f"{m:02d}" for m in range(1, 13)]

if latest_ym:
    default_year, default_month = latest_ym.split("-")
else:
    default_year, default_month = years_list[-1], months_list[-1]

with st.sidebar:
    st.header("Filter")
    col_year, col_month = st.columns(2)
    with col_year:
        year_input = st.selectbox(
            "년도",
            options=years_list,
            index=years_list.index(default_year) if default_year in years_list else len(years_list) - 1,
        )
    with col_month:
        month_input = st.selectbox(
            "월",
            options=months_list,
            index=months_list.index(default_month) if default_month in months_list else len(months_list) - 1,
        )

    year_month_input = f"{year_input}-{month_input}"


# ================= 기본 정보 + 최신 위험도 =================
summary = get_product_summary(asin, year_month=year_month_input)
if not summary or not summary.get("product"):
    st.error("해당 상품을 찾을 수 없습니다. parent_asin을 확인하세요.")
    st.stop()

product = summary["product"]
risk = summary.get("risk") or {}
metrics = summary.get("metrics") or {}



st.subheader(product.get("product_title", "-"))
st.markdown(
    f'<p class="caption-small">{product.get("store", "-")} · {product.get("main_category", "-")}</p>',
    unsafe_allow_html=True,
)


def render_kpi(col, label, value, delta=None, delta_color="normal", key=None):
    with col:
        with st.container(height=100, border=True, key=key):
            st.metric(label, value, delta=delta, delta_color=delta_color)


risk_prob = risk.get("risk_probability")

row1 = st.columns(7)
render_kpi(row1[0], "기준 월", summary.get("year_month", "-"), key="kpi-year-month")
render_kpi(row1[1], "평균 평점", metrics.get("avg_rating", "-"), key="kpi-avg-rating")
render_kpi(
    row1[2],
    "저평점 비율",
    f"{metrics.get('low_rating_ratio', 0):.1%}" if metrics.get("low_rating_ratio") is not None else "-",
    key="kpi-low-rating",
)
render_kpi(row1[3], "위험도", f"{risk_prob:.1%}" if risk_prob is not None else "데이터 없음", key="kpi-risk")
render_kpi(
    row1[4],
    "운영 신뢰도",
    f"{risk.get('operational_confidence', 0):.2f}" if risk.get("operational_confidence") is not None else "-",
    key="kpi-conf-score",
)
render_kpi(row1[5], "신뢰도 등급", risk.get("confidence_level", "-"), key="kpi-conf-level")
render_kpi(
    row1[6],
    "현재 리뷰 수",
    risk.get("current_review_count", metrics.get("review_count", "-")),
    key="kpi-review-count",
)


st.divider()


# ================= 월별 추이 =================
st.subheader("월별 위험도 / 평점 / 저평점 비율 추이", help="상품의 시간에 따른 지표 변화를 보여줍니다.")
trend = get_product_trend(asin)
if trend:
    trend_df = pd.DataFrame(trend).sort_values("year_month")
    metric_cols = [c for c in trend_df.columns if c != "year_month"]
    chosen = st.multiselect(
        "표시할 지표 선택",
        options=metric_cols,
        default=[c for c in ["risk_probability", "avg_rating", "low_rating_ratio"] if c in metric_cols],
    )
    if chosen:
        st.line_chart(trend_df.set_index("year_month")[chosen])
    st.dataframe(trend_df, use_container_width=True, hide_index=True)
else:
    st.info("추이 데이터가 없습니다.")

st.divider()


# ================= 최근 증가한 불만 토픽 =================
st.subheader("최근 증가한 불만 토픽", help="상품×월 리뷰에서 매칭된 불만 토픽별 언급 비율과 변화량입니다.")
topics, topics_ym = get_product_topics(asin, year_month=year_month_input)
if topics:
    st.markdown(f'<p class="caption-small">기준 월: {topics_ym}</p>', unsafe_allow_html=True)
    topics_df = pd.DataFrame(topics)
    st.dataframe(
        topics_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "topic_name": "토픽",
            "mention_count": "언급 수",
            "rate_t": st.column_config.NumberColumn("이번 달 비율", format="%.2f"),
            "rate_p3": st.column_config.NumberColumn("직전 3개월 비율", format="%.2f"),
            "delta": st.column_config.NumberColumn("변화량", format="%.2f"),
        },
    )

    chart_df = topics_df.sort_values("delta", ascending=False)
    fig = px.bar(chart_df, x="topic_name", y="delta", text="delta")
    fig.update_traces(marker_color="#FF9900")
    fig.update_layout(xaxis_title=None, yaxis_title="변화량")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("토픽 데이터가 없습니다.")

st.divider()


# ================= 위험 판단 근거 =================
st.subheader("위험 판단 근거")
explanation = get_risk_explanation(asin)
if explanation:
    if explanation.get("note"):
        st.warning(explanation["note"])
    exp_items = explanation.get("explanations")
    if exp_items:
        st.dataframe(
            pd.DataFrame(exp_items),
            use_container_width=True,
            hide_index=True,
            column_config={"feature_name": "변수명", "contribution": "기여도"},
        )
    else:
        st.info("위험 판단 근거 데이터가 없습니다.")
else:
    st.info("위험 판단 근거 데이터가 없습니다.")

st.divider()


# ================= 근거 리뷰 =================
st.subheader("근거 리뷰 (1~2점)", help="위험 판단의 근거가 되는 저평점 리뷰 원문입니다.")
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    max_rating = st.selectbox("최대 별점", options=[2, 1], index=0)
with fcol2:
    year_month_filter = st.text_input(
        "연-월 필터 (예: 2022-06, 선택)",
        value=year_month_input or "",
        help="비워두면 전체 기간에서 조회합니다.",
    )
with fcol3:
    topic_filter = st.selectbox(
        "토픽 필터",
        options=["전체"] + VALID_TOPICS,
        index=0,
    )
    topic_filter = None if topic_filter == "전체" else topic_filter

reviews = get_evidence_reviews(
    asin,
    max_rating=max_rating,
    year_month=year_month_filter or None,
    topic=topic_filter or None,
)
if reviews:
    st.dataframe(
        pd.DataFrame(reviews),
        use_container_width=True,
        hide_index=True,
        column_config={
            "rating": "별점",
            "review_text": "리뷰 내용",
            "related_topics": "관련 토픽",
            "review_datetime_utc": "작성일",
            "helpful_vote": "도움됨 수",
        },
    )
else:
    st.info("조건에 맞는 근거 리뷰가 없습니다.")


# ================= API 원본 =================
with st.expander("summary API 원본"):
    st.json(summary)