# ****************************************************************************
# (C) Cloudera, Inc. 2020-2026
# ****************************************************************************

import os
import json
import tempfile
from typing import TypedDict, List

import gradio as gr

from cdepy import (
    cdeconnection,
    cdemanager,
    cdejob,
    cderesource
)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END

# =========================================================
# CONFIGURATION (ENV VARS)
# =========================================================

JOBS_API_URL = os.environ["JOBS_API_URL"]
WORKLOAD_USER = os.environ["WORKLOAD_USER"]
WORKLOAD_PASSWORD = os.environ["WORKLOAD_PASSWORD"]

LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
LLM_ENDPOINT_BASE_URL = os.environ["LLM_ENDPOINT_BASE_URL"]
LLM_CDP_TOKEN = os.environ["LLM_CDP_TOKEN"]

RESOURCE_NAME = "llm-failing-spark-scripts"
JOB_PREFIX = "llm-failing-job"

# =========================================================
# CDE INIT
# =========================================================

CDE_CONNECTION = cdeconnection.CdeConnection(
    JOBS_API_URL,
    WORKLOAD_USER,
    WORKLOAD_PASSWORD,
)
CDE_CONNECTION.setToken()
CDE_MANAGER = cdemanager.CdeClusterManager(CDE_CONNECTION)

# =========================================================
# UI CACHE (STATIC)
# =========================================================

UI_RESULTS = {
    "status": "Not started",
    "resource_name": "",
    "jobs": [],
}

# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(TypedDict):
    scripts: List[dict]
    resource_created: bool

# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0,
)

# =========================================================
# LLM
# =========================================================

SPARK_APP_TEMPLATE = '''
# === BEGIN SPARK TEMPLATE (DO NOT REMOVE OR ABBREVIATE) ===

from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
import sys
import random
import numpy as np
from dbldatagen import DataGenerator
from datetime import date

# Setup
targetTable = sys.argv[1]
sourceTable = sys.argv[2]

today = date.today()

spark = SparkSession.builder \
    .appName(f"IcebergDynamicMultiKeySkew_{today}") \
    .getOrCreate()

row_count = int(np.random.normal(loc=50_000_000, scale=50_000_000))
row_count = max(row_count, 10_000_000)
print(f"Generating {row_count:,} rows")

def create_skew_ids(num_keys=10):
    ids = random.sample(range(10_000_000, 5_000_000_000), num_keys)
    weights = np.random.dirichlet(np.ones(num_keys), size=1)[0]
    return ids, weights

skewed_ids, skew_weights = create_skew_ids(num_keys=10)

overlap_fraction = random.uniform(0.1, 0.9)
overlap_count = int(len(skewed_ids) * overlap_fraction)
overlap_ids = skewed_ids[:overlap_count]
new_ids = [i + 10_000_000_000 for i in skewed_ids[overlap_count:]]
df2_ids = overlap_ids + new_ids
df2_weights = np.random.dirichlet(np.ones(len(df2_ids)), size=1)[0]

num_partitions = max(row_count // 1_000_000, 4)

df2_spec = (
    DataGenerator(spark, name="df2_gen", rows=row_count, partitions=num_partitions, seedColumnName="_seed_id")
    .withColumn("id", "long", values=df2_ids, weights=df2_weights)
    .withColumn("category", "string", values=["A", "B", "C", "D", "E"], random=True)
    .withColumn("value1", "double", minValue=0, maxValue=1000, random=True)
    .withColumn("value2", "double", minValue=0, maxValue=100, random=True)
    .withColumn("value3", "double", minValue=0, maxValue=1000, random=True)
    .withColumn("value4", "double", minValue=0, maxValue=100, random=True)
    .withColumn("value5", "double", minValue=0, maxValue=1000, random=True)
    .withColumn("value6", "double", minValue=0, maxValue=100, random=True)
    .withColumn("value7", "double", minValue=0, maxValue=1000, random=True)
    .withColumn("value8", "double", minValue=0, maxValue=100, random=True)
    .withColumn("event_ts", "timestamp", begin="2020-01-01 01:00:00", interval="1 day", random=True)
)

df2 = df2_spec.build().drop("_seed_id")

table_exists = spark._jsparkSession.catalog().tableExists(targetTable)

if not table_exists:
    df1_spec = (
        DataGenerator(spark, name="df1_gen", rows=row_count, partitions=num_partitions, seedColumnName="_seed_id")
        .withColumn("id", "long", values=skewed_ids, weights=skew_weights)
        .withColumn("category", "string", values=["A", "B", "C", "D", "E"], random=True)
        .withColumn("value1", "double", minValue=0, maxValue=1000, random=True)
        .withColumn("value2", "double", minValue=0, maxValue=100, random=True)
        .withColumn("value3", "double", minValue=0, maxValue=1000, random=True)
        .withColumn("value4", "double", minValue=0, maxValue=100, random=True)
        .withColumn("value5", "double", minValue=0, maxValue=1000, random=True)
        .withColumn("value6", "double", minValue=0, maxValue=100, random=True)
        .withColumn("value7", "double", minValue=0, maxValue=1000, random=True)
        .withColumn("value8", "double", minValue=0, maxValue=100, random=True)
        .withColumn("event_ts", "timestamp", begin="2020-01-01 01:00:00", interval="1 day", random=True)
    )

    df1 = df1_spec.build().drop("_seed_id")
    df1.writeTo(targetTable).using("iceberg").create()

spark.sql(f"DROP TABLE IF EXISTS {sourceTable} PURGE")
df2.writeTo(sourceTable).using("iceberg").create()

spark.sql(f\"\"\"
    MERGE INTO {targetTable} AS target
    USING {sourceTable} AS source
    ON target.id = source.id
    WHEN MATCHED AND source.event_ts > target.event_ts THEN
      UPDATE SET *
    WHEN NOT MATCHED THEN
      INSERT *
\"\"\")

spark.stop()

# === END SPARK TEMPLATE (DO NOT REMOVE OR ABBREVIATE) ===
'''


# =========================================================
# LANGGRAPH NODES
# =========================================================
import base64

def llm_generate_scripts(state: AgentState) -> AgentState:
    """
    Ask the LLM to produce 5 failing variants of a REAL Spark application
    by modifying Spark logic only. Uses base64 to avoid JSON corruption.
    """

    spark_template = SPARK_APP_TEMPLATE

    prompt = [
        SystemMessage(
            content=(
                "You are a senior Spark + Iceberg engineer.\n\n"
                "You are given a COMPLETE production PySpark application.\n"
                "Treat it as IMMUTABLE SOURCE CODE.\n\n"

                "TASK:\n"
                "Create EXACTLY 5 VARIANTS of this application.\n\n"

                "ABSOLUTE RULES:\n"
                "- You MUST return the ENTIRE script for each variant\n"
                "- You MUST NOT remove or summarize any code\n"
                "- You MUST keep BEGIN/END template markers\n"
                "- You MAY ONLY modify Spark / Iceberg logic\n"
                "- NO artificial Python errors\n\n"

                "FAILURE TYPES (use each once):\n"
                "1. Spark SQL / Catalyst analysis failure\n"
                "2. Iceberg schema mismatch during write or MERGE\n"
                "3. Runtime failure from extreme skew or repartitioning\n"
                "4. Ambiguous or invalid column resolution in MERGE\n"
                "5. Invalid MERGE semantics\n\n"

                "OUTPUT FORMAT (STRICT JSON ONLY):\n\n"
                "{\n"
                "  \"scripts\": [\n"
                "    {\n"
                "      \"name\": \"string\",\n"
                "      \"description\": \"why this Spark job fails\",\n"
                "      \"code_base64\": \"BASE64 ENCODED FULL SCRIPT\"\n"
                "    }\n"
                "  ]\n"
                "}\n\n"

                "IMPORTANT:\n"
                "- Encode the FULL SCRIPT using base64\n"
                "- Do NOT return raw code\n"
                "- Do NOT include markdown or explanations\n\n"

                "BASE APPLICATION:\n\n"
                f"{spark_template}"
            )
        )
    ]

    response = llm.invoke(prompt)

    # ---- Parse JSON safely ----
    payload = json.loads(response.content)

    if "scripts" not in payload or len(payload["scripts"]) != 5:
        raise RuntimeError("LLM did not return exactly 5 Spark scripts")

    import base64
    import re

    def validate_and_decode(original: str, encoded: str) -> str:
        # ---- Normalize Base64 (CRITICAL FIX) ----
        encoded = re.sub(r"\s+", "", encoded)  # remove newlines/spaces

        # Restore missing padding if needed
        missing_padding = len(encoded) % 4
        if missing_padding:
            encoded += "=" * (4 - missing_padding)

        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Base64 decode failed: {e}")

        # ---- Structural validation ----
        required_markers = [
            "BEGIN SPARK TEMPLATE",
            "END SPARK TEMPLATE",
            "DataGenerator",
            "MERGE INTO",
            "spark.stop()",
        ]

        for marker in required_markers:
            if marker not in decoded:
                raise RuntimeError(f"Missing required marker: {marker}")

        orig_lines = [l for l in original.splitlines() if l.strip()]
        gen_lines = [l for l in decoded.splitlines() if l.strip()]

        if len(gen_lines) < 0.9 * len(orig_lines):
            raise RuntimeError("Generated script was abbreviated")

        return decoded


def create_resource_once(state: AgentState) -> AgentState:
    """
    Ensure the CDE Files Resource exists.
    Idempotent: 409 (already exists) is treated as success.
    """

    if state.get("resource_created"):
        return state

    resource = cderesource.CdeFilesResource(RESOURCE_NAME)

    try:
        CDE_MANAGER.createResource(resource.createResourceDefinition())
        state["resource_status"] = "created"
        print(f"[CDE] Resource '{RESOURCE_NAME}' created")

    except Exception as e:
        msg = str(e)

        # ---- CRITICAL: Treat 409 as success ----
        if "already exists" in msg or "409" in msg:
            state["resource_status"] = "already_exists"
            print(f"[CDE] Resource '{RESOURCE_NAME}' already exists — continuing")
        else:
            # Real failure → stop graph
            raise RuntimeError(f"Failed to create resource '{RESOURCE_NAME}': {e}")

    state["resource_created"] = True
    return state


def upload_scripts(state: AgentState) -> AgentState:
    """
    Upload all generated scripts into the single resource.
    """
    for script in state["scripts"]:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False
        ) as f:
            f.write(script["code"])
            local_path = f.name

        CDE_MANAGER.uploadFileToResource(
            RESOURCE_NAME,
            os.path.dirname(local_path),
            os.path.basename(local_path),
        )

        script["filename"] = os.path.basename(local_path)

    return state


def create_and_run_jobs(state: AgentState) -> AgentState:
    """
    Create and run one Spark job per script.
    """
    jobs_for_ui = []

    for idx, script in enumerate(state["scripts"], start=1):
        job_name = f"{JOB_PREFIX}-{idx}"

        spark_job = cdejob.CdeSparkJob(CDE_CONNECTION)
        job_def = spark_job.createJobDefinition(
            CDE_JOB_NAME=job_name,
            CDE_RESOURCE_NAME=RESOURCE_NAME,
            APPLICATION_FILE_NAME=script["filename"],
            executorMemory="2g",
            executorCores=2,
        )

        CDE_MANAGER.createJob(job_def)
        CDE_MANAGER.runJob(job_name)

        jobs_for_ui.append({
            "job_name": job_name,
            "script": script["filename"],
            "failure_type": script["description"],
        })

    # Populate UI cache
    UI_RESULTS["status"] = "Jobs submitted"
    UI_RESULTS["resource_name"] = RESOURCE_NAME
    UI_RESULTS["jobs"] = jobs_for_ui

    return state

# =========================================================
# LANGGRAPH DEFINITION
# =========================================================

graph = StateGraph(AgentState)

graph.add_node("llm_generate_scripts", llm_generate_scripts)
graph.add_node("create_resource_once", create_resource_once)
graph.add_node("upload_scripts", upload_scripts)
graph.add_node("create_and_run_jobs", create_and_run_jobs)

graph.set_entry_point("llm_generate_scripts")

graph.add_edge("llm_generate_scripts", "create_resource_once")
graph.add_edge("create_resource_once", "upload_scripts")
graph.add_edge("upload_scripts", "create_and_run_jobs")
graph.add_edge("create_and_run_jobs", END)

app = graph.compile()

# =========================================================
# GRADIO CALLBACK
# =========================================================

def launch_pipeline_and_get_status():
    UI_RESULTS["status"] = "Launching..."

    app.invoke(
        {
            "scripts": [],
            "resource_created": False,
        }
    )

    jobs_text = "\n".join(
        f"- {j['job_name']} | Script: {j['script']} | Failure: {j['failure_type']}"
        for j in UI_RESULTS["jobs"]
    )

    return (
        UI_RESULTS["status"],
        UI_RESULTS["resource_name"],
        jobs_text,
    )

# =========================================================
# GRADIO UI (CLOUDERA AI FRIENDLY)
# =========================================================

css = """
.page-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 16px;
    text-align: center;
}
"""

with gr.Blocks(
    title="LLM-Driven Failing CDE Spark Jobs",
    css=css
) as demo:

    gr.Markdown(
        "<div class='page-title'>LLM-Generated Failing Spark Jobs (CDE)</div>"
    )

    gr.Markdown(
        "This app uses **LangGraph + an LLM** to generate **five intentionally failing "
        "PySpark applications**, upload them to **CDE**, and submit them as Spark jobs."
    )

    with gr.Row():
        launch_btn = gr.Button("🚀 Launch Failing Jobs")

    with gr.Row():
        status_box = gr.Textbox(
            label="Pipeline Status",
            interactive=False,
        )
        resource_box = gr.Textbox(
            label="CDE Files Resource",
            interactive=False,
        )

    jobs_box = gr.Textbox(
        label="Submitted Jobs",
        lines=10,
        interactive=False,
    )

    launch_btn.click(
        fn=launch_pipeline_and_get_status,
        inputs=[],
        outputs=[status_box, resource_box, jobs_box],
    )

# =========================================================
# MAIN (Cloudera AI Style)
# =========================================================

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT", "7860")),
    )
