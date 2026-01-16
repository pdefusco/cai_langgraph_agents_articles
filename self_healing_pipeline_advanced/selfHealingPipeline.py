#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#****************************************************************************

import json
import time
import threading
import os
import difflib
from typing import TypedDict

import gradio as gr

from cdepy import (
    cdeconnection,
    cdemanager,
    cdejob,
    cderesource
)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

import gradio as gr
print("Gradio runtime version:", gr.__version__)


# =========================================================
# USER CONFIGURATION
# =========================================================

JOBS_API_URL = os.environ["JOBS_API_URL"]
WORKLOAD_USER = os.environ["WORKLOAD_USER"]
WORKLOAD_PASSWORD = os.environ["WORKLOAD_PASSWORD"]

JOB_NAME = os.environ["JOB_NAME"]
RESOURCE_NAME = os.environ["RESOURCE_NAME"]
APPLICATION_FILE_NAME = os.environ["APPLICATION_FILE_NAME"]

LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
LLM_ENDPOINT_BASE_URL = os.environ["LLM_ENDPOINT_BASE_URL"]
LLM_CDP_TOKEN = os.environ["LLM_CDP_TOKEN"]

POLL_INTERVAL_SECONDS = 30

LAST_REMEDIATION_INFO = {
    "summary": "",
    "job_name": "",
    "resource_name": "",
}

# =========================================================
# GLOBAL CDE OBJECTS
# =========================================================

CDE_CONNECTION = None
CDE_MANAGER = None


def init_cde():
    global CDE_CONNECTION, CDE_MANAGER
    CDE_CONNECTION = cdeconnection.CdeConnection(
        JOBS_API_URL,
        WORKLOAD_USER,
        WORKLOAD_PASSWORD,
    )
    CDE_CONNECTION.setToken()
    CDE_MANAGER = cdemanager.CdeClusterManager(CDE_CONNECTION)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(TypedDict):
    latest_run_id: str | None
    latest_run_status: str | None

    spark_logs: str | None
    spark_script: str | None

    llm_analysis: str | None
    improved_script: str | None
    code_diff: str | None

    remediation_summary: str | None
    new_job_name: str | None
    new_resource_name: str | None

    retried: bool


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
# GRAPH NODES
# =========================================================

import os
import json

JOB_NAME = os.environ.get("JOB_NAME", "failing-pipeline")

def fetch_latest_run(state: dict = None) -> dict:
    """
    Fetch the latest run for the job matching JOB_NAME.
    """
    if state is None:
        state = {}  # Initialize empty dict if None

    # Get all job runs
    result = CDE_MANAGER.listJobRuns()
    if result == -1 or not result:
        state["latest_run_id"] = None
        state["latest_run_status"] = None
        return state

    try:
        runs = json.loads(result).get("runs", [])
    except Exception as e:
        raise RuntimeError(f"Failed to parse listJobRuns() response: {result}") from e

    # Filter runs by exact JOB_NAME
    job_runs = [r for r in runs if r.get("job") == JOB_NAME]

    if not job_runs:
        state["latest_run_id"] = None
        state["latest_run_status"] = None
        return state

    # Pick the latest run based on 'started' timestamp
    latest = max(job_runs, key=lambda r: r.get("started", ""))
    state["latest_run_id"] = str(latest.get("id"))
    state["latest_run_status"] = latest.get("status", "").upper()

    return state


def route_on_status(state: AgentState):
    if state["latest_run_status"] == "FAILED" and not state["retried"]:
        return "download_artifacts"
    return END


def download_artifacts(state: AgentState):
    run_id = state["latest_run_id"]

    logs = CDE_MANAGER.downloadJobRunLogs(run_id, "driver/stdout")
    state["spark_logs"] = logs or "No driver stdout logs available"

    print("\n========== DRIVER STDOUT LOGS ==========")
    print(state["spark_logs"])
    print("========== END DRIVER STDOUT LOGS ==========\n")

    script = CDE_MANAGER.downloadFileFromResource(
        RESOURCE_NAME,
        APPLICATION_FILE_NAME,
    )
    state["spark_script"] = script or ""
    return state


import difflib
import re

def llm_analyze_and_fix(state: AgentState):
    prompt = [
        SystemMessage(
            content=(
                "You are a senior Spark engineer.\n"
                "1. Identify the root cause of failure\n"
                "2. Explain why it happened\n"
                "3. Mention alternatives if relevant\n"
                "4. Produce a **complete, ready-to-run Spark Python script**.\n\n"
                "Respond EXACTLY in this format:\n"
                "=== ANALYSIS ===\n"
                "<analysis>\n\n"
                "=== FIXED SCRIPT ===\n"
                "<full python script, no placeholders, no backticks>"
            )
        ),
        HumanMessage(
            content=(
                f"SPARK SCRIPT:\n{state['spark_script']}\n\n"
                f"DRIVER STDOUT LOGS:\n{state['spark_logs']}"
            )
        ),
    ]

    response = llm.invoke(prompt)
    text = response.content

    # Split analysis and script
    if "=== FIXED SCRIPT ===" in text:
        analysis, fixed_script = text.split("=== FIXED SCRIPT ===", 1)
    else:
        # fallback if LLM didn't produce the exact separator
        analysis = "No analysis returned."
        fixed_script = state["spark_script"]

    # Remove any remaining backticks or markdown
    fixed_script = fixed_script.replace("```python", "").replace("```", "").strip()

    # Optionally remove placeholder comments
    lines = []
    for line in fixed_script.splitlines():
        if "# ... " in line or "# **FIX**" in line:
            continue  # skip illustrative comments
        if "spark.sql(f\"...\"" in line:
            continue  # skip placeholder SQL
        lines.append(line)
    fixed_script = "\n".join(lines)

    # Generate diff
    diff = difflib.unified_diff(
        state["spark_script"].splitlines(),
        fixed_script.splitlines(),
        fromfile="original.py",
        tofile="fixed.py",
        lineterm="",
    )

    state["llm_analysis"] = analysis.replace("=== ANALYSIS ===", "").strip()
    state["improved_script"] = fixed_script
    state["code_diff"] = "\n".join(diff)
    state["retried"] = True
    return state


import tempfile
import os

def deploy_and_run_fixed_job(state: AgentState):
    new_resource = f"{RESOURCE_NAME}-fixed"
    new_job_name = f"{JOB_NAME}-fixed"

    CDE_RESOURCE = cderesource.CdeFilesResource(new_resource)
    cdeFilesResourceDefinition = CDE_RESOURCE.createResourceDefinition()

    CDE_MANAGER.createResource(cdeFilesResourceDefinition)

    # ✅ WRITE SCRIPT TO LOCAL FILE FIRST
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False
    ) as f:
        f.write(state["improved_script"])
        local_path = f.name

    local_dir = os.path.dirname(local_path)
    local_file = os.path.basename(local_path)

    # ✅ NOW upload using cdepy expectations
    CDE_MANAGER.uploadFileToResource(
        new_resource,
        local_dir,
        local_file,
    )

    spark_job = cdejob.CdeSparkJob(CDE_CONNECTION)
    job_def = spark_job.createJobDefinition(
        new_job_name,
        new_resource,
        local_file,
        executorMemory="2g",
        executorCores=2,
        pythonEnvResourceName="datagen-env",
        args=["spark_catalog.default.dynamic_incremental_target_table_large_overlap",
                "spark_catalog.default.dynamic_incremental_source_table_large_overlap"]
    )

    CDE_MANAGER.createJob(job_def)
    CDE_MANAGER.runJob(new_job_name)

    summary = (
        f"New job created: {new_job_name}\n"
        f"New resource created: {new_resource}\n"
        f"Script uploaded: {APPLICATION_FILE_NAME}\n"
        f"Job submitted successfully."
    )

    state["remediation_summary"] = summary
    state["new_job_name"] = new_job_name
    state["new_resource_name"] = new_resource

    # ✅ GLOBAL CACHE FOR UI
    global LAST_REMEDIATION_INFO
    LAST_REMEDIATION_INFO = {
        "summary": summary,
        "job_name": new_job_name,
        "resource_name": new_resource,
    }

    return state


# =========================================================
# LANGGRAPH
# =========================================================

graph = StateGraph(AgentState)

graph.add_node("fetch_latest_run", fetch_latest_run)
graph.add_node("download_artifacts", download_artifacts)
graph.add_node("llm_analyze_and_fix", llm_analyze_and_fix)
graph.add_node("deploy_and_run_fixed_job", deploy_and_run_fixed_job)

graph.set_entry_point("fetch_latest_run")

graph.add_conditional_edges(
    "fetch_latest_run",
    route_on_status,
    {"download_artifacts": "download_artifacts", END: END},
)

graph.add_edge("download_artifacts", "llm_analyze_and_fix")
graph.add_edge("llm_analyze_and_fix", "deploy_and_run_fixed_job")
graph.add_edge("deploy_and_run_fixed_job", END)

app = graph.compile()


# =========================================================
# AGENT LOOP
# =========================================================

def run_monitor():
    app.invoke(
        {
            "latest_run_id": None,
            "latest_run_status": None,
            "spark_logs": None,
            "spark_script": None,
            "llm_analysis": None,
            "improved_script": None,
            "code_diff": None,
            "new_job_name": None,
            "new_resource_name": None,
            "retried": False,
        }
    )


def agent_loop():
    while True:
        run_monitor()
        time.sleep(POLL_INTERVAL_SECONDS)


# =========================================================
# UI
# =========================================================

def ui_refresh(state: dict = None):
    state = state or {}
    state = fetch_latest_run(state)
    latest_run_id = state.get("latest_run_id")
    latest_run_status = state.get("latest_run_status", "UNKNOWN")

    spark_script = ""
    spark_logs = ""
    llm_analysis = ""
    improved_script = ""
    code_diff = ""

    if latest_run_id:
        try:
            spark_logs = CDE_MANAGER.downloadJobRunLogs(str(latest_run_id), "driver/stdout") or ""
            spark_script = CDE_MANAGER.downloadFileFromResource(RESOURCE_NAME, APPLICATION_FILE_NAME) or ""

            state["spark_script"] = spark_script
            state["spark_logs"] = spark_logs

            if not state.get("retried", False) and latest_run_status == "FAILED":
                state = llm_analyze_and_fix(state)

            llm_analysis = state.get("llm_analysis", "")
            improved_script = state.get("improved_script", "")
            code_diff = state.get("code_diff", "")
        except Exception as e:
            llm_analysis = f"Failed to fetch logs or script: {e}"

    status_text = (
        f"Job Name: {JOB_NAME}\n"
        f"Latest Run ID: {latest_run_id or 'N/A'}\n"
        f"Status: {latest_run_status}\n"
        f"Jobs API URL: {JOBS_API_URL}\n"
        f"Application File: {APPLICATION_FILE_NAME}"
    )

    remediation_summary_text = LAST_REMEDIATION_INFO.get("summary", "No remediation info yet.")

    updated_job_text = (
        f"Job Name: {LAST_REMEDIATION_INFO.get('job_name', 'N/A')}\n"
        f"Resource Name: {LAST_REMEDIATION_INFO.get('resource_name', 'N/A')}\n"
        f"Application File: {APPLICATION_FILE_NAME}"
    )

    return (
        status_text,
        remediation_summary_text,
        updated_job_text,
        spark_script,
        spark_logs,
        llm_analysis,
        improved_script,
        code_diff
    )


# =========================================================
# STARTUP
# =========================================================

init_cde()

def start_agent():
    threading.Thread(target=agent_loop, daemon=True).start()

css = """
/* Page title at the top */
.page-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 16px;
    text-align: center;
}

/* Titles inside status/remediation boxes */
.status-box-title {
    font-weight: bold;
    font-size: 16px;
    margin-bottom: 4px;
    color: #333;
}

/* Boxes themselves */
.status-box {
    border: 1px solid #ccc;
    padding: 8px;
    border-radius: 5px;
    background-color: #f9f9f9;
}

/* Scrollable code for spark scripts */
.scrollable-code .gr-code {
    max-height: 300px;
    overflow-y: auto;
}
"""

with gr.Blocks(title="CDE Spark Job Monitor & Auto Remediator", css=css) as demo:

    # Page title
    gr.Markdown("<div class='page-title'>CDE Spark Job Monitor & Auto Remediator</div>")

    # Top row: original job status, remediation summary, updated job info
    with gr.Row():
        status_box = gr.Textbox(
            label="Original Job Status",
            lines=5,
            interactive=False
        )
        remediation_summary_box = gr.Textbox(
            label="Remediation Summary",
            lines=5,
            interactive=False
        )
        updated_job_box = gr.Textbox(
            label="Updated Job (Remediated) Information",
            lines=5,
            interactive=False
        )

    # Second row: code, logs, analysis
    with gr.Row():
        script_box = gr.Code(
            label="Spark Script",
            language="python",
            show_label=True,
            interactive=False,
            elem_classes=["scrollable-code"]
        )
        fixed_script_box = gr.Code(
            label="Improved Spark Script",
            language="python",
            show_label=True,
            interactive=False,
            elem_classes=["scrollable-code"]
        )

    # Third row: logs, LLM analysis, code diff
    with gr.Row():
        logs_box = gr.Textbox(
            label="Driver Stdout Logs",
            lines=15,
            max_lines=15
        )
        analysis_box = gr.Textbox(
            label="LLM Analysis (Root Cause & Explanation)",
            lines=10,
            max_lines=10
        )
        diff_box = gr.Textbox(
            label="Spark Code Diff (Original vs Fixed)",
            lines=20,
            max_lines=20
        )

    # Timer to refresh UI every 10 seconds
    timer = gr.Timer(value=10, active=True)  # refresh every 2 minutes
    timer.tick(
        fn=ui_refresh,
        inputs=[],
        outputs=[
            status_box,
            remediation_summary_box,
            updated_job_box,
            script_box,
            logs_box,
            analysis_box,
            fixed_script_box,
            diff_box,
        ]
    )

    demo.load(fn=start_agent)


if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT")),
    )
