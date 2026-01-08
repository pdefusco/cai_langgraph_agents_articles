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
# Shared UI State
# ============================================================

from copy import deepcopy
from threading import Lock

class UIState(TypedDict):
    last_app_id: Optional[str]
    last_job_launch_time: Optional[str]
    anomalies: List[dict]
    tuning_recommendations: List[dict]
    last_updated: Optional[str]

# ======= Initialize UI_STATE =======
UI_STATE = {
    "last_app_id": "Waiting for Spark metrics...",
    "last_job_launch_time": "N/A",
    "anomaly_md": "No anomalies detected yet.",
    "tuning_md": "No tuning recommendations yet.",
    "last_updated": "N/A",
}
UI_STATE_LOCK = threading.Lock()


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
        print("ROW SANITY CHECK: ", row)
        if row.get("shuffleBytesWritten", 0) > 100:
            anomalies.append({
                "spark_application_id": str(row["appId"]),
                "metric": "shuffleBytesWritten",
                "value": float(row["shuffleBytesWritten"]),
                "threshold": 100,
                "severity": "high",
            })

        jvm = row.get("jvmGCTime")
        run = row.get("executorRunTime")
        if run and run > 0:
            gc_pct = float(jvm / run * 100)
            if gc_pct > 20:
                anomalies.append({
                    "spark_application_id": str(row["appId"]),
                    "metric": "gc_time_pct",
                    "value": gc_pct,
                    "threshold": 20,
                    "severity": "medium",
                })

    print("ANOMALIES THIS INVOCATION: ", anomalies)
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

        results = spark_col.query(query_embeddings=[emb], n_results=5)

        spark_opts = []
        for d in results["documents"][0]:
            if "spark.sql.shuffle.partitions" in d:
                spark_opts.append("--conf spark.sql.shuffle.partitions=2000")
            if "spark.executor.memoryOverhead" in d:
                spark_opts.append("--conf spark.executor.memoryOverhead=2g")

        tuning_recommendations.append({
            "spark_application_id": str(anomaly["spark_application_id"]),
            "issue": str(anomaly["metric"]),
            "recommended_spark_submit": [str(x) for x in spark_opts],
        })

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

def format_anomalies(anomalies: list[dict]) -> str:
    if not anomalies:
        return "✅ No anomalies detected."

    lines = ["### 🚨 Detected Anomalies"]
    for a in anomalies:
        lines.append(
            f"- **App** `{a['spark_application_id']}` | "
            f"**Metric** `{a['metric']}` | "
            f"**Value** `{a['value']:.2f}` | "
            f"**Severity** `{a['severity']}`"
        )
    return "\n".join(lines)


def format_tuning(recs: list[dict]) -> str:
    if not recs:
        return "ℹ️ No tuning recommendations."

    lines = ["### 🛠️ Tuning Recommendations"]
    for r in recs:
        opts = ", ".join(r["recommended_spark_submit"]) or "None"
        lines.append(
            f"- **App** `{r['spark_application_id']}` | "
            f"**Issue** `{r['issue']}`\n"
            f"  - Suggested flags: `{opts}`"
        )
    return "\n".join(lines)

def agent_loop():
    create_checkpoint(spark)
    last_launch_time, _ = load_checkpoint(spark)

    while True:
        where_clause = ""
        if last_launch_time:
            where_clause = f"WHERE TIMESTAMP(ts) > TIMESTAMP('{last_launch_time}')"

        start = time.time()
        # Aggregate metrics per Spark application
        df = spark.sql(f"""
            SELECT
                appId,
                MIN(ts) AS first_ts,
                MAX(ts) AS last_ts,
                SUM(shuffleBytesWritten) AS shuffleBytesWritten,
                SUM(executorRunTime) AS executorRunTime,
                SUM(jvmGCTime) AS jvmGCTime
            FROM {SPARK_METRICS_TABLE}
            {where_clause}
            GROUP BY appId
            ORDER BY last_ts
        """)

        row_count = df.count()
        print(f"ROWS FETCHED: {row_count}")
        df.show(5, truncate=False)
        duration = time.time() - start
        print(f"Query returned {row_count} rows in {duration:.2f}s")

        # Convert all rows to dicts
        rows = [r.asDict() for r in df.collect()] if row_count > 0 else []

        if rows:
            # Take the latest application for checkpointing and graph invoke
            latest_app_row = rows[-1]
            latest_app_id = latest_app_row["appId"]
            last_ts = latest_app_row["last_ts"]

            print("LAST PROCESSED:", last_launch_time, latest_app_id)

            # Optionally: pass all rows to graph.invoke if you want anomalies per app
            result = graph.invoke({
                "metrics": rows,  # all aggregated rows
                "agent_version": AGENT_VERSION,
            })
            print("GRAPH OUTPUT:", result)

            # Save checkpoint for the latest processed app
            save_checkpoint(spark, last_ts, latest_app_id)
        else:
            latest_app_id = "N/A"
            last_ts = last_launch_time or "N/A"
            result = {"anomalies": [], "tuning_recommendations": []}

        # Update UI_STATE with all aggregated metrics and markdowns
        with UI_STATE_LOCK:
            UI_STATE["last_app_id"] = str(latest_app_id)
            UI_STATE["last_job_launch_time"] = str(last_ts)
            UI_STATE["anomaly_md"] = format_anomalies(result.get("anomalies", []))
            UI_STATE["tuning_md"] = format_tuning(result.get("tuning_recommendations", []))
            UI_STATE["last_updated"] = datetime.utcnow().isoformat()

            # Keep all aggregated metrics as a Pandas DataFrame
            UI_STATE["aggregated_metrics_df"] = df.toPandas() if row_count > 0 else None

        time.sleep(POLL_INTERVAL_SECONDS)


# ============================================================
# Gradio UI
# ============================================================

def get_ui_state():
    with UI_STATE_LOCK:
        s = deepcopy(UI_STATE)

    # Convert aggregated metrics dataframe to HTML for display
    metrics_df = s.get("aggregated_metrics_df")
    if metrics_df is not None:
        metrics_html = metrics_df.to_html(index=False)
    else:
        metrics_html = "No Spark metrics available"

    return (
        metrics_html,
        s.get("anomaly_md", "✅ No anomalies detected"),
        s.get("tuning_md", "ℹ️ No tuning recommendations"),
        s.get("last_updated", ""),
        s.get("last_app_id", ""),
        s.get("last_job_launch_time", ""),
    )


# ============================================================
# Gradio UI
# ============================================================

def start_agent():
    """Starts the background agent loop in a daemon thread."""
    threading.Thread(target=agent_loop, daemon=True).start()
    # No return needed; UI updates are handled by the timer


# Initialize UI_STATE placeholders (optional but recommended)
with UI_STATE_LOCK:
    UI_STATE.setdefault("last_app_id", "Waiting for Spark metrics...")
    UI_STATE.setdefault("last_job_launch_time", "")
    UI_STATE.setdefault("anomaly_md", "Waiting for anomalies...")
    UI_STATE.setdefault("tuning_md", "Waiting for tuning recommendations...")
    UI_STATE.setdefault("last_updated", "")


with gr.Blocks(title="Spark Performance Monitoring Agent") as demo:
    gr.Markdown("## 🔍 Spark Performance Monitoring Agent")

    metrics_table = gr.HTML(label="Spark Metrics by App")
    anomalies = gr.Markdown(label="Detected Anomalies")
    tuning = gr.Markdown(label="Tuning Recommendations")
    updated = gr.Textbox(label="Last Updated (UTC)")
    last_app = gr.Textbox(label="Last Spark Application ID")
    last_launch = gr.Textbox(label="Last Job Launch Time")

    timer = gr.Timer(value=10, active=True)
    timer.tick(
        fn=get_ui_state,
        inputs=[],
        outputs=[metrics_table, anomalies, tuning, updated, last_app, last_launch],
    )

    demo.load(fn=start_agent)
    #demo.load(fn=get_ui_state, outputs=[last_app, last_launch, anomalies, tuning, updated, metrics_table])

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo.launch(share=False,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
