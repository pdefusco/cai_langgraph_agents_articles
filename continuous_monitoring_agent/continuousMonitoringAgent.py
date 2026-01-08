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
from copy import deepcopy
import json
import os
import gradio as gr
from pyspark.sql import SparkSession
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import chromadb
from chromadb.config import Settings

# ============================================================
# Spark session
# ============================================================

spark = (
    SparkSession.builder
    .appName("spark-performance-agent-ui")
    .config("spark.executor.cores", 2)
    .config("spark.execuctor.memory", "4g")
    .config("spark.kerberos.access.hadoopFileSystems","s3a://pdf-jan-26-buk-7c0e831f/")
    .getOrCreate()
)

# ============================================================
# Constants
# ============================================================

AGENT_NAME = "spark_perf_agent"
AGENT_VERSION = "v2"
POLL_INTERVAL_SECONDS = 30

SPARK_METRICS_TABLE = "default.spark_task_metrics"

# ============================================================
# Shared UI State
# ============================================================

class UIState(TypedDict):
    anomaly_md: str
    tuning_md: str
    last_updated: Optional[str]
    aggregated_metrics_df: Optional[object]

UI_STATE: UIState = {
    "anomaly_md": "No anomalies detected yet.",
    "tuning_md": "No tuning recommendations yet.",
    "last_updated": "N/A",
    "aggregated_metrics_df": None,
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
# Node 1: Deterministic anomaly detection
# ============================================================

def analyze_metrics(state: AgentState) -> AgentState:
    anomalies = []

    for row in state["metrics"]:
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

    return {"anomalies": anomalies}

# ============================================================
# Chroma + LLM setup
# ============================================================
import re
import json

def extract_json(text: str):
    if not text:
        raise ValueError("Empty LLM response")

    # Remove code fences
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    # Extract first JSON array or object
    match = re.search(r"(\[.*?\]|\{.*?\})", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM output")

    json_text = match.group(1)

    # Remove JS-style comments
    json_text = re.sub(r"//.*", "", json_text)

    return json.loads(json_text)


LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
LLM_ENDPOINT_BASE_URL = os.environ["LLM_ENDPOINT_BASE_URL"]
LLM_CDP_TOKEN = os.environ["LLM_CDP_TOKEN"]

client = chromadb.PersistentClient()
spark_col = client.get_or_create_collection("spark_tuning")

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0.2,
)

# ============================================================
# Embedding helpers (unchanged on purpose)
# ============================================================
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
EMBEDDING_ENDPOINT_BASE_URL = os.environ["EMBEDDING_ENDPOINT_BASE_URL"]
EMBEDDING_CDP_TOKEN = os.environ["EMBEDDING_CDP_TOKEN"]

from openai import OpenAI

llmClient = OpenAI(
    base_url=EMBEDDING_ENDPOINT_BASE_URL,
    api_key=EMBEDDING_CDP_TOKEN,
)

def get_passage_embedding(text: str):
    return llmClient.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL_ID,
        extra_body={"input_type": "passage"},
    ).data[0].embedding

# ============================================================
# Improved RAG query construction
# ============================================================

def build_rag_query(anomaly: dict) -> str:
    metric = anomaly["metric"]

    if metric == "shuffleBytesWritten":
        return (
            "Spark tuning for excessive shuffle data, large shuffles, "
            "shuffle spill to disk, shuffle partitions, wide transformations"
        )

    if metric == "gc_time_pct":
        return (
            "Spark tuning for high JVM garbage collection time, "
            "executor memory pressure, GC overhead, executor sizing"
        )

    return (
        "Spark performance tuning for executor memory, "
        "parallelism, and shuffle behavior"
    )

# ============================================================
# LLM-based Spark recommendation
# ============================================================

def llm_recommend_spark_configs(anomaly: dict, docs: list[str]) -> list[dict]:
    system = SystemMessage(
        content=(
            "You are a Spark performance tuning expert. "
            "Analyze Spark performance anomalies and documentation excerpts. "
            "Recommend concrete Spark configuration changes. "
            "Be concise and practical."
        )
    )

    human = HumanMessage(
        content=f"""
Detected anomaly:
- Metric: {anomaly['metric']}
- Value: {anomaly['value']}
- Severity: {anomaly['severity']}

Relevant Spark documentation excerpts:
{chr(10).join(docs)}

Return ONLY valid JSON in this format:
[
  {{
    "config": "spark.sql.shuffle.partitions",
    "value": "2000",
    "reason": "Large shuffle volume benefits from higher parallelism"
  }}
]
"""
    )

    response = llm.invoke([system, human])
    raw = response.content
    print("LLM RAW OUTPUT:\n", raw)
    return extract_json(response.content)

# ============================================================
# Node 2: RAG tuning (LLM-driven)
# ============================================================

def rag_tuning(state: AgentState) -> AgentState:
    tuning_recommendations = []

    for anomaly in state["anomalies"]:
        query = build_rag_query(anomaly)
        emb = get_passage_embedding(query)

        results = spark_col.query(
            query_embeddings=[emb],
            n_results=5,
        )

        docs = results["documents"][0]

        try:
            llm_recs = llm_recommend_spark_configs(anomaly, docs)
        except Exception as e:
            print("LLM recommendation failed:", e)
            llm_recs = []

        spark_submit_flags = [
            f"--conf {r['config']}={r['value']}"
            for r in llm_recs
        ]

        tuning_recommendations.append({
            "spark_application_id": anomaly["spark_application_id"],
            "issue": anomaly["metric"],
            "recommended_spark_submit": spark_submit_flags,
            "details": llm_recs,
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
# Formatting helpers
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
        lines.append(
            f"\n#### 📌 App `{r['spark_application_id']}`"
            f"\n- **Issue**: `{r['issue']}`"
        )

        # Spark-submit flags
        if r.get("recommended_spark_submit"):
            lines.append("\n**Suggested spark-submit flags:**")
            for opt in r["recommended_spark_submit"]:
                lines.append(f"- `{opt}`")
        else:
            lines.append("\n_No spark-submit flags generated._")

        # Detailed LLM reasoning
        details = r.get("details", [])
        if details:
            lines.append("\n**Why these changes:**")
            for d in details:
                lines.append(
                    f"- **{d['config']} = {d['value']}**  \n"
                    f"  _Reason_: {d['reason']}"
                )

    return "\n".join(lines)

# ============================================================
# Agent loop
# ============================================================

def agent_loop():
    while True:
        df = spark.sql(f"""
            SELECT
                appId,
                MIN(ts) AS first_ts,
                MAX(ts) AS last_ts,
                SUM(shuffleBytesWritten) AS shuffleBytesWritten,
                SUM(executorRunTime) AS executorRunTime,
                SUM(jvmGCTime) AS jvmGCTime
            FROM {SPARK_METRICS_TABLE}
            GROUP BY appId
            ORDER BY last_ts
        """)

        rows = [r.asDict() for r in df.collect()] if df.count() > 0 else []

        if rows:
            result = graph.invoke({
                "metrics": rows,
                "agent_version": AGENT_VERSION,
            })
        else:
            result = {"anomalies": [], "tuning_recommendations": []}

        with UI_STATE_LOCK:
            UI_STATE["anomaly_md"] = format_anomalies(result.get("anomalies", []))
            UI_STATE["tuning_md"] = format_tuning(result.get("tuning_recommendations", []))
            UI_STATE["last_updated"] = datetime.utcnow().isoformat()
            UI_STATE["aggregated_metrics_df"] = df.toPandas() if rows else None

        time.sleep(POLL_INTERVAL_SECONDS)

# ============================================================
# Gradio UI
# ============================================================

def get_ui_state():
    with UI_STATE_LOCK:
        s = deepcopy(UI_STATE)

    metrics_df = s.get("aggregated_metrics_df")
    metrics_html = metrics_df.to_html(index=False) if metrics_df is not None else "No Spark metrics"

    return (
        metrics_html,
        s["anomaly_md"],
        s["tuning_md"],
        s["last_updated"],
    )

def start_agent():
    threading.Thread(target=agent_loop, daemon=True).start()

with gr.Blocks(title="Spark Performance Monitoring Agent") as demo:
    gr.Markdown("## 🔍 Spark Performance Monitoring Agent")

    metrics_table = gr.HTML(label="Spark Metrics by App")
    anomalies = gr.Markdown(label="Detected Anomalies")
    tuning = gr.Markdown(label="Tuning Recommendations")
    updated = gr.Textbox(label="Last Updated (UTC)")

    timer = gr.Timer(value=10, active=True)
    timer.tick(
        fn=get_ui_state,
        inputs=[],
        outputs=[metrics_table, anomalies, tuning, updated],
    )

    demo.load(fn=start_agent)

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT")),
    )
