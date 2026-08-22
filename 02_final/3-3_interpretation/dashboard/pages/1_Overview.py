import pandas as pd
import streamlit as st
import plotly.express as px

from utils.api_client import get_risky_products, search_products, get_risk_changes

st.set_page_config(page_title="전체 현황", layout="wide")


# ================= 커스텀 스타일 =================
st.markdown("""
<style>
h1 { font-size: 30px !important; font-weight: 700 !important; }
h3 { font-size: 20px !important; font-weight: 700 !important; }
[data-testid="stCaptionContainer"] { font-size: 20px; font-weight: 500; }


.caption-small { font-size: 15px; font-weight: 500; }
.product-name { font-size: 16px !important; font-weight: 700 !important; margin: 0; }
.product-meta { font-size: 14px !important; font-weight: 400 !important; color: #888; margin: 0; }

[data-testid="stMetricValue"] { font-size: 26px; font-weight: 600; }

[class*="st-key-kpi1"],
[class*="st-key-kpi2"],
[class*="st-key-kpi3"],
[class*="st-key-kpi4"] {
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
.product-name { font-size: 20px; font-weight: 700; margin: 0; }
.product-meta { font-size: 12px; font-weight: 400; color: #888; margin: 0; }

[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 6px 14px !important;
}

.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.logo("assets/amazon_logo.png", size="large")  

st.title("품질 위험 상품 모니터링")
st.caption("Appliances")


# ================= 사이드바 필터 =================
with st.sidebar:
    st.header("Filter")

    category_input = st.selectbox(
        "카테고리",
        options=["Appliances"],  
        index=0,
        disabled=False,  
        help="현재는 Appliances 카테고리만 지원 가능합니다.",
    )

    col_year, col_month = st.columns(2)
    with col_year:
        year_input = st.selectbox(
            "년도",
            options=["전체"] + [str(y) for y in range(2016, 2024)],
            index=0,
        )
    with col_month:
        month_input = st.selectbox(
            "월",
            options=["전체"] + [f"{m:02d}" for m in range(1, 13)],
            index=0,
        )

    if year_input == "전체" or month_input == "전체":
        year_month_input = ""
    else:
        year_month_input = f"{year_input}-{month_input}"

    sort_by = st.selectbox(
        "정렬 기준",
        options=["risk", "confidence_weighted"],
        format_func=lambda x: "위험도순" if x == "risk" else "위험도×운영신뢰도순",
    )
    limit = st.slider("표시 개수", min_value=10, max_value=100, value=50, step=10)
    high_risk_threshold = st.slider("고위험 기준(%)", min_value=50, max_value=95, value=70, step=5)

    confidence_level_input = st.multiselect(
        "신뢰도 등급",
        options=["높음", "보통"],  # 실제 DB 값에 맞게 조정 필요 (아래 확인 방법 참고)
        default=[],
        help="미선택시 전체 등급이 표기됩니다.",
    )
    brand_input = st.text_input("브랜드 검색")
    min_review_count = st.number_input("최소 리뷰 수", min_value=0, value=0, step=5)


# ================= 데이터 로드 =================
with st.spinner("데이터 불러오는 중..."):
    results, shown_year_month = get_risky_products(
        sort_by=sort_by, limit=limit, year_month=year_month_input or None
    )

if not results:
    st.info(
        f"'{year_month_input}' 월에는 데이터가 없습니다. 다른 월을 선택하거나 비워서 최신 월을 확인하세요."
        if year_month_input else
        "표시할 데이터가 없습니다. 백엔드 콘솔에 에러가 없는지, DB에 데이터가 있는지 확인하세요."
    )
    st.stop()

df = pd.DataFrame(results)
risk_changes_data = get_risk_changes(
    year_month=year_month_input or None,
    high_risk_threshold=high_risk_threshold / 100,
)

if confidence_level_input and "confidence_level" in df.columns:
    df = df[df["confidence_level"].isin(confidence_level_input)]

if brand_input and "store" in df.columns:
    df = df[df["store"].str.contains(brand_input, case=False, na=False)]

if min_review_count > 0 and "current_review_count" in df.columns:
    df = df[df["current_review_count"] >= min_review_count]

if df.empty:
    st.info("필터 조건에 맞는 상품이 없습니다. 필터를 조정해보세요.")
    st.stop()


# ================= KPI 카드 =================
total_count = len(df)
if "risk_probability" in df.columns:
    high_risk_count = (df["risk_probability"] >= high_risk_threshold / 100).sum()
    avg_risk = df["risk_probability"].mean()
else:
    high_risk_count, avg_risk = "-", None


kcol1, kcol2, kcol3, kcol4 = st.columns(4)
def render_kpi(col, label, value, delta=None, delta_color="normal", key=None):
    with col:
        with st.container(height=115, border=True, key=key):
            st.metric(label, value, delta=delta, delta_color=delta_color)

render_kpi(kcol1, "기준 월", shown_year_month or "검색 결과", key="kpi1")
render_kpi(kcol2, "조회된 상품 수", f"{total_count:,}", key="kpi2")

if risk_changes_data and risk_changes_data.get("high_risk_count_change") is not None:
    change = risk_changes_data["high_risk_count_change"]
    render_kpi(
        kcol3,
        f"고위험 상품 (≥{high_risk_threshold}%)",
        f"{high_risk_count:,}" if isinstance(high_risk_count, int) else high_risk_count,
        delta=f"{change:+d}",
        delta_color="inverse",
        key="kpi3"
    )
else:
    render_kpi(
        kcol3,
        f"고위험 상품 (≥{high_risk_threshold}%)",
        f"{high_risk_count:,}" if isinstance(high_risk_count, int) else high_risk_count,
        key="kpi3"
    )

render_kpi(kcol4, "평균 위험도", f"{avg_risk:.1%}" if avg_risk is not None else "-", key="kpi4")

st.divider()


# ================= 위험도 % 컬럼 준비 =================
if "risk_probability" in df.columns:
    df = df.rename(columns={"risk_probability": "risk_probability_pct"})
    df["risk_probability_pct"] = df["risk_probability_pct"] * 100



col_left, col_right = st.columns(2)


# ================= 왼쪽: 위험 등급별 상품 분포 =================
with col_left:
    st.subheader("위험 등급별 상품 분포")
    if "risk_probability_pct" in df.columns:
        def risk_bucket(p):
            if p >= 70: return "고위험 (70%+)"
            elif p >= 40: return "중위험 (40~70%)"
            else: return "저위험 (40% 미만)"

        bucket_series = df["risk_probability_pct"].apply(risk_bucket)  

        bucket_order = ["고위험 (70%+)", "중위험 (40~70%)", "저위험 (40% 미만)"]
        bucket_counts = (
            bucket_series.value_counts()   
            .reindex(bucket_order, fill_value=0)
            .reset_index()
        )
        bucket_counts.columns = ["등급", "상품 수"]

        color_map = {
            "고위험 (70%+)": "#FF9900",
            "중위험 (40~70%)": "#FF9900",
            "저위험 (40% 미만)": "#FF9900",
        }

        fig = px.bar(
            bucket_counts,
            x="등급",
            y="상품 수",
            color="등급",
            color_discrete_map=color_map,
            category_orders={"등급": bucket_order},
            text="상품 수",
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="상품 수")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("위험도 데이터가 없습니다.")


# ================= 오른쪽: 위험도 급상승 상품 =================
with col_right:
    st.subheader(
        "전월 대비 위험도 급상승 상품",
        help="필터와 무관하게 전체 카테고리 기준입니다.",
        )

    if risk_changes_data and risk_changes_data.get("products"):
        st.markdown(
            f'<p class="caption-small">{risk_changes_data["previous_year_month"]} → {risk_changes_data["year_month"]} </p>',
            unsafe_allow_html=True,
        )

        top_n = 3
        top_products = risk_changes_data["products"][:top_n]

        for rank, product in enumerate(top_products, start=1):
            change = product["risk_change"] * 100
            current = product["current_risk"] * 100
            previous = product["previous_risk"] * 100

            with st.container(border=True):
                col_info, col_metric = st.columns([4.2, 0.8])

                with col_info:
                    st.markdown(
                        f'<p class="product-name"><span class="rank-badge">#{rank}</span>{product["product_title"][:60]}</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<p class="product-meta">{product["store"]} · {product["parent_asin"]}</p>',
                        unsafe_allow_html=True,
                    )
                
                delta_color = "#D62728" if change >= 0 else "#2CA02C"
                delta_bg = "rgba(214, 39, 40, 0.12)" if change >= 0 else "rgba(44, 160, 44, 0.12)"
                arrow = "▲" if change >= 0 else "▼"

                with col_metric:
                    st.markdown(
                        f"""
                        <div class="risk-up-metric" style="border-left: 1px solid #E0E0E0; padding-left: 16px; padding-bottom: 10px;">
                            <p class="risk-up-delta">{current:.1f}%</p>
                            <p class="risk-up-value">
                                <span class="risk-up-pill" style="color:{delta_color}; background-color:{delta_bg};">
                                    {arrow} {abs(change):.1f}%p
                                </span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    
    else:
        st.info("비교할 이전 달 데이터가 없습니다.")

st.divider()


# ================= 상품 목록 =================
st.subheader("상품 목록",
        help="행을 클릭하면 상세 화면으로 이동합니다.",
)

column_config = {
    "risk_probability_pct": st.column_config.ProgressColumn(
        "위험도", format="%.1f%%", min_value=0, max_value=100
    ),
    "operational_confidence": st.column_config.NumberColumn("운영 신뢰도", format="%.2f"),
    "product_title": st.column_config.TextColumn("상품명", width="large"),
    "store": "브랜드/스토어",
    "current_review_count": "현재 리뷰 수",
    "confidence_level": "신뢰도 등급",
    "parent_asin": "ASIN",
}

event = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
    on_select="rerun",
    selection_mode="single-row",
)


# ================= 행 클릭 -> 상세 이동 =================
selected_rows = event.selection.rows if hasattr(event, "selection") else []
if selected_rows:
    selected_asin = df.iloc[selected_rows[0]]["parent_asin"]
    st.session_state["selected_asin"] = selected_asin
    st.switch_page("pages/2_Product_Detail.py")

st.divider()

# 폴백: 구버전 streamlit 등 클릭 선택이 안 되는 환경 대비
with st.expander("상품 선택"):
    fallback_asin = st.selectbox("상품 선택", options=df["parent_asin"].tolist())
    if st.button("상품 상세 화면으로 이동 →"):
        st.session_state["selected_asin"] = fallback_asin
        st.switch_page("pages/2_Product_Detail.py")