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

def fetch_latest_run(state: AgentState):
    # Fetch all job runs from CDE
    result = CDE_MANAGER.listJobRuns()
    if result == -1 or not result:
        # API failed or returned nothing
        state["latest_run_id"] = None
        state["latest_run_status"] = None
        return state

    try:
        runs = json.loads(result).get("runs", [])
    except Exception as e:
        raise RuntimeError(f"Failed to parse listJobRuns() response: {result}") from e

    # Filter runs by the configured JOB_NAME
    job_runs = [r for r in runs if r.get("job") == JOB_NAME]

    if not job_runs:
        state["latest_run_id"] = None
        state["latest_run_status"] = None
        return state

    # Pick the latest run based on the 'started' timestamp
    latest = max(job_runs, key=lambda r: r.get("started", ""))
    state["latest_run_id"] = str(latest.get("id"))  # ensure it's a string
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
                "4. Produce a corrected Spark script\n\n"
                "Respond EXACTLY in this format:\n"
                "=== ANALYSIS ===\n"
                "<analysis>\n\n"
                "=== FIXED SCRIPT ===\n"
                "<python code>"
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

    # Split the analysis and fixed script
    try:
        analysis, fixed_script = text.split("=== FIXED SCRIPT ===", 1)
    except ValueError:
        # If splitting fails, treat entire output as analysis
        analysis = text
        fixed_script = ""

    # Clean analysis and fixed script
    analysis = analysis.replace("=== ANALYSIS ===", "").strip()
    fixed_script = fixed_script.strip()

    # Strip Markdown code fences if present in fixed script
    fixed_script = re.sub(r"^```(?:python)?\n", "", fixed_script)  # remove opening ``` or ```python
    fixed_script = re.sub(r"\n```$", "", fixed_script)             # remove closing ```
    fixed_script = fixed_script.strip()

    # Generate diff safely
    diff = difflib.unified_diff(
        state["spark_script"].splitlines(),
        fixed_script.splitlines(),
        fromfile="original.py",
        tofile="fixed.py",
        lineterm="",
    )

    # Update state
    state["llm_analysis"] = analysis
    state["improved_script"] = fixed_script
    state["code_diff"] = "\n".join(diff)
    state["retried"] = True

    return state


import tempfile
import os

def deploy_and_run_fixed_job(state: AgentState):
    new_resource = f"{RESOURCE_NAME}_fixed"
    new_job_name = f"{JOB_NAME}_fixed"

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
    )

    CDE_MANAGER.createJob(job_def)
    CDE_MANAGER.runJob(new_job_name)

    # Optional cleanup
    #os.remove(local_path)

    state["remediation_summary"] = (
        f"Created resource '{new_resource}', "
        f"job '{new_job_name}', and submitted a new run."
    )

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

def ui_refresh():
    runs = json.loads(CDE_MANAGER.listJobRuns()).get("runs", [])
    if not runs:
        return "", "", "", "", "", "", ""

    latest = max(runs, key=lambda r: r.get("started", ""))
    run_id = str(latest.get("id"))

    status = latest.get("status", "").upper()
    script = CDE_MANAGER.downloadFileFromResource(
        RESOURCE_NAME, APPLICATION_FILE_NAME
    ) or ""

    logs = CDE_MANAGER.downloadJobRunLogs(run_id, "driver/stdout") or ""

    return status, script, logs, "", "", "", ""


# =========================================================
# STARTUP
# =========================================================

init_cde()
threading.Thread(target=agent_loop, daemon=True).start()

with gr.Blocks(title="CDE Spark Job Auto-Remediator") as demo:
    gr.Markdown("## CDE Spark Job Monitor & Auto-Remediator")

    status_box = gr.Textbox(label="Job Status")
    script_box = gr.Code(label="Original Spark Script", language="python")
    logs_box = gr.Textbox(label="Driver Stdout Logs", lines=15)

    analysis_box = gr.Textbox(
        label="LLM Analysis (Root Cause & Explanation)", lines=10
    )
    fixed_script_box = gr.Code(
        label="Improved Spark Script", language="python"
    )
    diff_box = gr.Textbox(
        label="Spark Code Diff (Original vs Fixed)",
        lines=20,
        interactive=False,
    )
    remediation_box = gr.Textbox(
        label="Remediation Summary", lines=4
    )

    refresh_btn = gr.Button("Refresh")
    refresh_btn.click(
        fn=ui_refresh,
        outputs=[
            status_box,
            script_box,
            logs_box,
            analysis_box,
            fixed_script_box,
            diff_box,
            remediation_box,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT")),
        show_error=True,
    )
