from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from pyhive import hive
import os

# =========================================================
# App
# =========================================================

app = FastAPI()

# =========================================================
# On-Prem Nemotron
# =========================================================

LLM = ChatOpenAI(
    model="nemotron-onprem",
    api_key=os.getenv("ON_PREM_NEMOTRON_KEY"),
    base_url=os.getenv("ON_PREM_NEMOTRON_ENDPOINT"),
)

# =========================================================
# Hive Query Helper
# =========================================================

def run_hive_query(sql: str) -> str:
    conn = hive.Connection(
        host=os.getenv("HIVE_HOST"),
        port=int(os.getenv("HIVE_PORT", "10000")),
        username=os.getenv("HIVE_USER", "hive"),
    )
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    return str(rows)

# =========================================================
# Agent Endpoint
# =========================================================

@app.post("/invoke")
async def invoke(payload: dict):
    question = payload["question"]

    prompt = f"""
You are a Text-to-SQL agent.

Convert the user question into a Hive SQL query.
Use ONLY this table:

sales_table(
    date STRING,
    region STRING,
    revenue DOUBLE
)

Question:
{question}

Return ONLY valid Hive SQL.
"""

    sql = LLM.invoke(prompt).content.strip()
    result = run_hive_query(sql)

    return {
        "sql": sql,
        "result": result
    }
