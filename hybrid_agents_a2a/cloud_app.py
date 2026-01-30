import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pyspark.sql import SparkSession
import uvicorn
import threading


# =========================================================
# FastAPI app
# =========================================================
app = FastAPI()

# =========================================================
# Contract / Access Control
# =========================================================
ALLOWED_TABLES = {"DataLakeEtl"}

# =========================================================
# On-prem Nemotron (OpenAI-compatible)
# =========================================================
LLM = ChatOpenAI(
    model=os.getenv("ON_PREM_MODEL_ID"),
    api_key=os.getenv("ON_PREM_MODEL_KEY"),
    base_url=os.getenv("ON_PREM_MODEL_ENDPOINT"),
    timeout=15,
    max_retries=1,
)


# =========================================================
# Spark Session (shared, created once)
# =========================================================
spark = (
    SparkSession.builder
    .appName("on-prem-text-to-sql-agent")
    .master("local[*]")
    .getOrCreate()
)

# Optional: limit runaway queries during demos
# spark.conf.set("spark.sql.shuffle.partitions", "10")
# spark.conf.set("spark.executor.cores", 4)
# spark.conf.set("spark.executor.memory", "8g")

# =========================================================
# Spark SQL Executor
# =========================================================
def run_spark_sql(sql: str) -> list[dict]:
    """
    Execute Spark SQL and return a safe, truncated JSON result.
    """
    sql = sql.strip().strip("`")
    df = spark.sql(sql)

    # Limit rows
    MAX_ROWS = 20
    df = df.limit(MAX_ROWS)

    # Optional: truncate columns to avoid huge payloads
    MAX_COLS = 8
    df = df.select(df.columns[:MAX_COLS])

    # Convert to list of dicts for JSON
    rows = df.toPandas().to_dict(orient="records")

    print(f"[Spark] Returning {len(rows)} rows with {len(df.columns)} columns")  # logging
    return rows


# =========================================================
# Enforce SQL Permissions
# =========================================================
def validate_sql_tables(sql: str, allowed_tables: set[str]):
    sql_lower = sql.lower()
    for table in allowed_tables:
        if table.lower() in sql_lower:
            return
    raise ValueError("SQL violates table access contract")

# =========================================================
# Health Endpoint
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================================================
# Agent Endpoint
# =========================================================
@app.post("/invoke")
def invoke(payload: dict):
    print(">>> invoke called")

    # -----------------------------
    # Validate contract
    # -----------------------------
    contract = payload.get("contract")
    if not contract:
        return {"error": "Missing contract"}

    requested_tables = set(contract.get("requested_tables", []))
    if not requested_tables:
        return {"error": "Contract must specify requested_tables"}

    unauthorized = requested_tables - ALLOWED_TABLES
    if unauthorized:
        return {
            "error": "Unauthorized table access",
            "unauthorized_tables": list(unauthorized),
        }

    # Enforce single-table access (optional but recommended)
    if len(requested_tables) != 1:
        return {"error": "Only one table may be requested"}

    table_name = next(iter(requested_tables))
    print(f">>> contract approved for table: {table_name}")

    # -----------------------------
    # Extract question
    # -----------------------------
    question = payload.get("request", {}).get("question", "")
    if not question:
        return {"error": "Missing question in payload"}

    print(">>> question:", question)

    # -----------------------------
    # Prompt (bound to contract)
    # -----------------------------
    prompt = f"""
You are a Text-to-SQL agent.

Your task:
- Convert the user's question into a Spark SQL query.
- The query will be executed using Spark SQL.
- Use ONLY the table described below.
- Do NOT hallucinate columns or tables.
- Do NOT include explanations.

Authorized Table (from contract):
{table_name}

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
- fraud_trx INT

Rules:
- Use valid Spark SQL
- Return ONLY the SQL statement

User question:
{question}
"""

    print(">>> calling LLM")
    sql = LLM.invoke(prompt).content.strip()
    try:
        validate_sql_tables(sql, {table_name})
    except ValueError:
        return {
            "error": "Generated SQL references a table not allowed by the contract",
            "allowed_tables": [table_name],
        }

    print(">>> generated SQL:", sql)

    print(">>> running spark")
    result = run_spark_sql(sql)
    print(">>> spark done")

    return {
        "sql": sql,
        "result": result,
        "table": table_name,
    }


# =========================================================
# Start the server (safe for Cloudera AI container)
# =========================================================
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']), log_level="warning", reload=False)

server_thread = threading.Thread(target=run_server)
server_thread.start()
