import pandas as pd
from config import engine

df = pd.read_sql(
    "SELECT parent_asin, risk_probability FROM product_month_risk WHERE `year_month`='2023-01' LIMIT 10",
    con=engine
)
print(df)
