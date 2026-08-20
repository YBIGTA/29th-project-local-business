DROP DATABASE IF EXISTS amazon_risk_service;
CREATE DATABASE amazon_risk_service DEFAULT CHARACTER SET utf8mb4;
USE amazon_risk_service;

-- 1. products : 상품 기본 정보
CREATE TABLE products (
parent_asin VARCHAR(20) NOT NULL,
product_title TEXT,
store VARCHAR(255),
main_category VARCHAR(100),
categories TEXT,
PRIMARY KEY (parent_asin)
);

-- 2. product_month_metrics : 상품x월 정형 지표
CREATE TABLE product_month_metrics (
parent_asin VARCHAR(20) NOT NULL,
`year_month` CHAR(7) NOT NULL,
review_count INT,
avg_rating FLOAT,
low_rating_count INT,
low_rating_ratio FLOAT,
verified_purchase_ratio FLOAT,
mean_helpful_vote FLOAT,
text_available_ratio FLOAT,
past_3m_review_count INT,
history_3m_low_rating_ratio FLOAT,
history_6m_low_rating_ratio FLOAT,
history_3m_vs_6m_low_rating_ratio_delta FLOAT,
history_12m_review_count INT,
history_12m_avg_rating FLOAT,
history_12m_low_rating_ratio FLOAT,
history_3m_vs_12m_low_rating_ratio_delta FLOAT,
is_review_volume_drop_2m TINYINT,
PRIMARY KEY (parent_asin, `year_month`),
FOREIGN KEY (parent_asin) REFERENCES products(parent_asin),
INDEX idx_low_rating_ratio (low_rating_ratio)
);

-- 3. product_month_risk : 상품x월 위험도/신뢰도
CREATE TABLE product_month_risk (
parent_asin VARCHAR(20) NOT NULL,
`year_month` CHAR(7) NOT NULL,
model_version VARCHAR(50) NOT NULL DEFAULT 's45_mil_v1',
risk_probability FLOAT,
current_review_count INT,
s45_raw_score FLOAT,
s45_raw_mil_score FLOAT,
mil_score_change FLOAT,
review_evidence_confidence FLOAT,
model_consistency_confidence FLOAT,
operational_confidence FLOAT,
confidence_level VARCHAR(20),
PRIMARY KEY (parent_asin, `year_month`, model_version),
FOREIGN KEY (parent_asin) REFERENCES products(parent_asin),
INDEX idx_risk_prob (risk_probability)
);

-- 4. product_month_topics : 상품x월x토픽 (긴 형태)
CREATE TABLE product_month_topics (
parent_asin VARCHAR(20) NOT NULL,
`year_month` CHAR(7) NOT NULL,
topic_name VARCHAR(60) NOT NULL,
mention_count INT,
rate_t FLOAT,
rate_p3 FLOAT,
delta FLOAT,
rate_t_shrunk FLOAT,
PRIMARY KEY (parent_asin, `year_month`, topic_name),
FOREIGN KEY (parent_asin) REFERENCES products(parent_asin),
INDEX idx_delta (delta)
);

-- 5. evidence_reviews : 위험 근거용 선별 리뷰
CREATE TABLE evidence_reviews (
review_id BIGINT AUTO_INCREMENT PRIMARY KEY,
parent_asin VARCHAR(20) NOT NULL,
`year_month` CHAR(7) NOT NULL,
rating TINYINT,
review_text TEXT,
related_topics VARCHAR(300),
review_datetime_utc DATETIME,
verified_purchase BOOLEAN,
helpful_vote INT DEFAULT 0,
FOREIGN KEY (parent_asin) REFERENCES products(parent_asin),
INDEX idx_asin_month (parent_asin, `year_month`),
INDEX idx_rating (rating)
);

-- 6. risk_explanations : SHAP 결과 (구조만 우선 생성)
CREATE TABLE risk_explanations (
parent_asin VARCHAR(20) NOT NULL,
`year_month` CHAR(7) NOT NULL,
feature_name VARCHAR(100) NOT NULL,
contribution FLOAT,
PRIMARY KEY (parent_asin, `year_month`, feature_name),
FOREIGN KEY (parent_asin) REFERENCES products(parent_asin)
);
