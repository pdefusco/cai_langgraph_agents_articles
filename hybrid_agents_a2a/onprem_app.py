import os
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pyspark.sql import SparkSession

# =========================================================
# FastAPI app
# =========================================================

app = FastAPI()

# =========================================================
# On-prem Nemotron (OpenAI-compatible)
# =========================================================

LLM = ChatOpenAI(
    model=os.getenv("ON_PREM_MODEL_ID"),
    api_key=os.getenv("ON_PREM_MODEL_KEY"),
    base_url=os.getenv("ON_PREM_MODEL_ENDPOINT"),
)

# =========================================================
# Spark Session (shared, created once)
# =========================================================

spark = (
    SparkSession.builder
    .appName("on-prem-text-to-sql-agent")
    .enableHiveSupport()   # IMPORTANT for external / metastore tables
    .getOrCreate()
)

# Optional: limit runaway queries during demos
spark.conf.set("spark.sql.shuffle.partitions", "10")

# =========================================================
# Spark SQL Executor
# =========================================================

def run_spark_sql(sql: str) -> str:
    """
    Execute Spark SQL and return a small, safe string result.
    """
    df = spark.sql(sql)

    # Limit output size defensively
    rows = df.limit(20).toPandas()

    return rows.to_string(index=False)

# =========================================================
# Agent Endpoint
# =========================================================

@app.post("/invoke")
async def invoke(payload: dict):
    question = payload["question"]

    prompt = f"""
You are a Text-to-SQL agent.

Your task:
- Convert the user's question into a Spark SQL query.
- The query will be executed using Spark SQL.
- Use ONLY the table described below.
- Do NOT hallucinate columns or tables.
- Do NOT include explanations.

Table:
DataLakeTable

Schema:
- age FLOAT
- credit_card_balance FLOAT
- bank_account_balance FLOAT
- mortgage_balance FLOAT
- sec_bank_account_balance FLOAT
- savings_account_balance FLOAT
- sec_savings_account_balance FLOAT
- total_est_nworth FLOAT
- primary_loan_balance FLOAT
- secondary_loan_balance FLOAT
- uni_loan_balance FLOAT
- longitude FLOAT
- latitude FLOAT
- transaction_amount FLOAT
- fraud_trx INT   -- 1 = fraud, 0 = non-fraud

Rules:
- Use valid Spark SQL
- Return ONLY the SQL statement

User question:
{question}
"""

    sql = LLM.invoke(prompt).content.strip()

    result = run_spark_sql(sql)

    return {
        "sql": sql,
        "result": result
    }
