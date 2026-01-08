#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

"""
Continuous Spark Performance Monitoring Agent with Gradio UI
"""

import time
import threading
from datetime import datetime
from typing import List, Optional, TypedDict
import gradio as gr
from pyspark.sql import SparkSession
from langgraph.graph import StateGraph, END
import chromadb
from chromadb.config import Settings

# ============================================================
# Spark session
# ============================================================

spark = (
    SparkSession.builder
    .appName("spark-performance-agent-ui")
    .config("spark.executor.cores", 2)
    .config("spark.execuctor.memory", '4g')
    .config("spark.kerberos.access.hadoopFileSystems","s3a://pdf-jan-26-buk-7c0e831f/")
    .getOrCreate()
)

# ============================================================
# Constants
# ============================================================

AGENT_NAME = "spark_perf_agent"
AGENT_VERSION = "v1"
POLL_INTERVAL_SECONDS = 30

SPARK_METRICS_TABLE = "default.spark_task_metrics"
CHECKPOINT_TABLE = "agent_checkpoints"

# ============================================================
# Shared UI State (thread-safe enough for this use case)
# ============================================================

UI_STATE = {
    "last_app_id": None,
    "last_job_launch_time": None,
    "anomalies": [],
    "tuning_recommendations": [],
    "last_updated": None,
}

# ============================================================
# LangGraph State
# ============================================================

class AgentState(TypedDict):
    metrics: List[dict]
    anomalies: Optional[List[dict]]
    tuning_recommendations: Optional[List[dict]]
    agent_version: str

# ============================================================
# Checkpoint helpers
# ============================================================

def create_checkpoint(spark):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CHECKPOINT_TABLE} (
            agent_name STRING,
            agent_version STRING,
            last_job_launch_time STRING,
            last_spark_application_id STRING,
            updated_at TIMESTAMP
        )
        PARTITIONED BY (agent_name, agent_version)
        STORED AS PARQUET
    """)

def load_checkpoint(spark):
    df = spark.sql(f"""
        SELECT last_job_launch_time, last_spark_application_id
        FROM {CHECKPOINT_TABLE}
        WHERE agent_name = '{AGENT_NAME}'
          AND agent_version = '{AGENT_VERSION}'
        ORDER BY updated_at DESC
        LIMIT 1
    """)

    if df.count() == 0:
        return None, None

    r = df.collect()[0]
    return r.last_job_launch_time, r.last_spark_application_id


def save_checkpoint(spark, launch_time, app_id):
    spark.sql(f"""
        INSERT INTO {CHECKPOINT_TABLE}
        PARTITION (agent_name='{AGENT_NAME}', agent_version='{AGENT_VERSION}')
        VALUES (
            TIMESTAMP('{launch_time}'),
            '{app_id}',
            TIMESTAMP('{datetime.utcnow()}')
        )
    """)

# ============================================================
# Node 1: Deterministic anomaly detection
# ============================================================

def analyze_metrics(state: AgentState) -> AgentState:
    anomalies = []

    for row in state["metrics"]:
        UI_STATE["last_app_id"] = row["appId"]
        UI_STATE["last_job_launch_time"] = row["launchTime"]

        if row.get("shuffleBytesWritten", 0) > 1024:
            anomalies.append({
                "spark_application_id": row["appId"],
                "metric": "shuffleBytesWritten",
                "value": row["shuffleBytesWritten"],
                "threshold": 1024,
                "severity": "high",
            })

        jvmGCTime = row.get("jvmGCTime")
        executorRunTime = row.get("executorRunTime")
        gc_time_pct = (row["jvmGCTime"] / row["executorRunTime"]) * 100

        if gc_time_pct > 20:
            anomalies.append({
                "spark_application_id": row["appId"],
                "metric": "gc_time_pct",
                "value": gc_time_pct,
                "threshold": 20,
                "severity": "medium",
            })

    UI_STATE["anomalies"] = anomalies
    UI_STATE["last_updated"] = datetime.utcnow().isoformat()

    return {"anomalies": anomalies}

# ============================================================
# Chroma setup
# ============================================================

# multi_agent_langgraph_demo.py
import requests
import chromadb
from bs4 import BeautifulSoup
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
import gradio as gr
import time
import os

# -------------------------
# 1️⃣ Initialize clients
# -------------------------
MODEL_ID = os.environ["MODEL_ID"]
ENDPOINT_BASE_URL = os.environ["ENDPOINT_BASE_URL"]
CDP_TOKEN = os.environ["CDP_TOKEN"]

client = chromadb.PersistentClient()

# -------------------------
# 2️⃣ Scrape helper
# -------------------------
def fetch_text(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.find_all(["p", "li", "h1","h2","h3","pre"])
    return "\n".join([p.get_text(separator=" ", strip=True) for p in paragraphs])

def chunk_text(text, max_len=250):
    lines = text.split("\n")
    chunks, current = [], []
    for line in lines:
        current.append(line)
        if sum(len(l) for l in current) > max_len:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

import requests
from openai import OpenAI
import json
from tenacity import retry, wait_exponential, stop_after_attempt

llmClient = OpenAI(
    base_url=ENDPOINT_BASE_URL,  # EXACT value from AIS UI (ends with /v1)
    api_key=CDP_TOKEN,
)

def get_query_embedding(text: str):
    return llmClient.embeddings.create(
        input=text,
        model=MODEL_ID,
        extra_body={"input_type": "query"},
    ).data[0].embedding

def get_passage_embedding(text: str):
    return llmClient.embeddings.create(
        input=text,
        model=MODEL_ID,
        extra_body={"input_type": "passage"},
    ).data[0].embedding

# -------------------------
# 3️⃣ Chroma collections
# -------------------------
spark_col = client.get_or_create_collection("spark_tuning")
hadoop_col = client.get_or_create_collection("hadoop_perf")

# -------------------------
# 4️⃣ Chroma ingestion
# -------------------------
def ingest_demo_data():
    # Spark docs
    spark_url = "https://spark.apache.org/docs/latest/tuning.html"
    spark_chunks = chunk_text(fetch_text(spark_url))
    for i, chunk in enumerate(spark_chunks):
        spark_col.add(
            ids=[f"spark_{i}"],  # <--- unique ID for each chunk
            documents=[chunk],
            metadatas=[{"source": "spark_tuning", "chunk_index": i}],
            embeddings=[get_passage_embedding(chunk)]  # match the retrieval method
        )
        time.sleep(0.5)

    # Hadoop docs
    hadoop_url = "https://openlogic.com/blog/how-to-improve-hadoop-performance"
    hadoop_chunks = chunk_text(fetch_text(hadoop_url))
    for j, chunk in enumerate(hadoop_chunks):
        hadoop_col.add(
            ids=[f"hadoop_{j}"],  # <--- unique ID for each chunk
            documents=[chunk],
            metadatas=[{"source": "hadoop_perf", "chunk_index": j}],
            embeddings=[get_passage_embedding(chunk)]  # match retrieval
        )
        time.sleep(0.5)

ingest_demo_data()
print("Spark docs:", len(spark_col.get()["documents"]))
print("Hadoop docs:", len(hadoop_col.get()["documents"]))

ISSUE_TO_QUERY = {
    "shuffle_spill_mb": "Spark shuffle spill tuning",
    "gc_time_pct": "Spark GC overhead tuning",
}

# ============================================================
# Node 2: RAG tuning
# ============================================================

def rag_tuning(state: AgentState) -> AgentState:
    tuning_recommendations = []

    for anomaly in state["anomalies"]:
        query = ISSUE_TO_QUERY.get(anomaly["metric"], "Spark performance tuning")
        emb = get_passage_embedding(query)

        results = spark_col.query(
            query_embeddings=[emb],
            n_results=5,
        )

        spark_opts = []

        docs = results["documents"][0]
        for d in docs:
            if "spark.sql.shuffle.partitions" in d:
                spark_opts.append("--conf spark.sql.shuffle.partitions=2000")
            if "spark.executor.memoryOverhead" in d:
                spark_opts.append("--conf spark.executor.memoryOverhead=2g")

        tuning_recommendations.append({
            "spark_application_id": anomaly["spark_application_id"],
            "issue": anomaly["metric"],
            "recommended_spark_submit": list(set(spark_opts)),
        })

    UI_STATE["tuning_recommendations"] = tuning_recommendations
    UI_STATE["last_updated"] = datetime.utcnow().isoformat()

    return {"tuning_recommendations": tuning_recommendations}

# ============================================================
# Routing
# ============================================================

def route(state: AgentState):
    return "rag_tuning" if state.get("anomalies") else END

# ============================================================
# Build graph
# ============================================================

builder = StateGraph(AgentState)
builder.add_node("analyze", analyze_metrics)
builder.add_node("rag_tuning", rag_tuning)
builder.set_entry_point("analyze")

builder.add_conditional_edges(
    "analyze",
    route,
    {"rag_tuning": "rag_tuning", END: END},
)

builder.add_edge("rag_tuning", END)

graph = builder.compile()

# ============================================================
# Polling loop (background thread)
# ============================================================

def agent_loop():
    create_checkpoint(spark)
    last_launch_time, last_app_id = load_checkpoint(spark)

    while True:
        where_clause = ""
        if last_launch_time:
            where_clause = f"""
            WHERE (
                job_launch_time > TIMESTAMP('{last_launch_time}')
                OR (
                    job_launch_time = TIMESTAMP('{last_launch_time}')
                    AND spark_application_id > '{last_app_id}'
                )
            )
            """

        df = spark.sql(f"""
            SELECT *
            FROM {SPARK_METRICS_TABLE}
            {where_clause}
            ORDER BY launchTime, appId
        """)

        if df.count() > 0:
            rows = [r.asDict() for r in df.collect()]
            graph.invoke({"metrics": rows, "agent_version": AGENT_VERSION})

            last = rows[-1]
            save_checkpoint(spark, last["launchTime"], last["appId"])
            last_launch_time = last["launchTime"]
            last_app_id = last["appId"]

        time.sleep(POLL_INTERVAL_SECONDS)

# ============================================================
# Gradio UI
# ============================================================

def get_ui_state():
    return (
        UI_STATE["last_app_id"],
        UI_STATE["last_job_launch_time"],
        UI_STATE["anomalies"],
        UI_STATE["tuning_recommendations"],
        UI_STATE["last_updated"],
    )

with gr.Blocks(title="Spark Performance Monitoring Agent") as demo:
    gr.Markdown("## 🔍 Spark Performance Monitoring Agent")

    with gr.Row():
        last_app = gr.Textbox(label="Last Spark Application ID")
        last_launch = gr.Textbox(label="Last Job Launch Time")

    anomalies = gr.JSON(label="Detected Anomalies")
    tuning = gr.JSON(label="Tuning Recommendations")
    updated = gr.Textbox(label="Last Updated (UTC)")

    timer = gr.Timer(value=10, active=True)
    timer.tick(
        fn=get_ui_state,
        inputs=[],
        outputs=[last_app, last_launch, anomalies, tuning, updated]
    )

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    threading.Thread(target=agent_loop, daemon=True).start()
    demo.queue(default_concurrency_limit=16).launch(share=False,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
