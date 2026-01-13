#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#****************************************************************************

import json
import time
import threading
import os
from typing import TypedDict

import gradio as gr

from cdepy import (
    cdeconnection,
    cdemanager,
    cdejob,
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

    print("Initializing CDE connection...")
    CDE_CONNECTION = cdeconnection.CdeConnection(
        JOBS_API_URL,
        WORKLOAD_USER,
        WORKLOAD_PASSWORD,
    )
    CDE_CONNECTION.setToken()
    CDE_MANAGER = cdemanager.CdeClusterManager(CDE_CONNECTION)
    print("CDE initialized successfully")


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(TypedDict):
    latest_run_id: str | None
    latest_run_status: str | None
    spark_logs: str | None
    spark_script: str | None
    improved_script: str | None
    retried: bool


# =========================================================
# LLM (ONLY FOR SCRIPT FIX)
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
    print("[fetch_latest_run] Listing job runs...")
    result = CDE_MANAGER.listJobRuns()

    if result == -1 or not result:
        raise RuntimeError("listJobRuns() failed")

    runs = json.loads(result).get("runs", [])
    if not runs:
        state["latest_run_id"] = None
        state["latest_run_status"] = None
        return state

    latest = max(runs, key=lambda r: r.get("started", ""))
    state["latest_run_id"] = str(latest.get("id"))
    state["latest_run_status"] = latest.get("status", "").upper()

    print(
        f"[fetch_latest_run] Latest run: "
        f"id={state['latest_run_id']} "
        f"status={state['latest_run_status']}"
    )

    return state


def route_on_status(state: AgentState):
    if state["latest_run_status"] == "FAILED" and not state["retried"]:
        print("[route_on_status] Job FAILED → downloading artifacts")
        return "download_artifacts"

    print("[route_on_status] No action required")
    return END


def download_artifacts(state: AgentState):
    run_id = state["latest_run_id"]

    if not run_id:
        raise RuntimeError("No run_id available for log download")

    print(f"[download_artifacts] Downloading driver stdout logs for run {run_id}")

    logs = CDE_MANAGER.downloadJobRunLogs(run_id, "driver/stdout")

    if logs == -1 or logs is None:
        state["spark_logs"] = (
            f"ERROR: Failed to download driver stdout logs for run {run_id}"
        )
    elif isinstance(logs, str) and logs.strip():
        state["spark_logs"] = logs
    else:
        state["spark_logs"] = (
            f"No driver stdout logs returned for run {run_id}"
        )

    ### NEW: PRINT LOGS TO SCREEN
    print("\n========== DRIVER STDOUT LOGS ==========")
    print(state["spark_logs"])
    print("========== END DRIVER STDOUT LOGS ==========\n")

    print("[download_artifacts] Downloading Spark script...")
    script = CDE_MANAGER.downloadFileFromResource(
        RESOURCE_NAME,
        APPLICATION_FILE_NAME,
    )

    if script is None or script == -1:
        raise RuntimeError("Failed to download Spark script")

    state["spark_script"] = script
    return state


def llm_fix_script(state: AgentState):
    print("[llm_fix_script] Sending script + logs to LLM")

    prompt = [
        SystemMessage(
            content=(
                "You are a senior Spark engineer.\n"
                "Analyze the Spark driver stdout logs and fix the application.\n"
                "Preserve original intent and structure.\n"
                "Return ONLY valid Python Spark code."
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

    state["improved_script"] = response.content
    state["retried"] = True

    print("[llm_fix_script] LLM produced updated script")
    return state


def deploy_and_run_fixed_job(state: AgentState):
    new_resource = f"{RESOURCE_NAME}_fixed"
    new_job_name = f"{JOB_NAME}_fixed"

    print(f"[deploy_and_run_fixed_job] Creating resource {new_resource}")
    CDE_MANAGER.createResource(new_resource)

    print("[deploy_and_run_fixed_job] Uploading improved script")
    CDE_MANAGER.uploadFileToResource(
        new_resource,
        APPLICATION_FILE_NAME,
        state["improved_script"],
    )

    spark_job = cdejob.CdeSparkJob(CDE_CONNECTION)
    job_def = spark_job.createJobDefinition(
        new_job_name,
        new_resource,
        APPLICATION_FILE_NAME,
        executorMemory="2g",
        executorCores=2,
    )

    print(f"[deploy_and_run_fixed_job] Creating and running job {new_job_name}")
    CDE_MANAGER.createJob(job_def)
    CDE_MANAGER.runJob(new_job_name)

    return state


# =========================================================
# LANGGRAPH DEFINITION
# =========================================================

graph = StateGraph(AgentState)

graph.add_node("fetch_latest_run", fetch_latest_run)
graph.add_node("download_artifacts", download_artifacts)
graph.add_node("llm_fix_script", llm_fix_script)
graph.add_node("deploy_and_run_fixed_job", deploy_and_run_fixed_job)

graph.set_entry_point("fetch_latest_run")

graph.add_conditional_edges(
    "fetch_latest_run",
    route_on_status,
    {
        "download_artifacts": "download_artifacts",
        END: END,
    },
)

graph.add_edge("download_artifacts", "llm_fix_script")
graph.add_edge("llm_fix_script", "deploy_and_run_fixed_job")
graph.add_edge("deploy_and_run_fixed_job", END)

app = graph.compile()


# =========================================================
# AGENT LOOP
# =========================================================

def run_monitor():
    try:
        app.invoke(
            {
                "latest_run_id": None,
                "latest_run_status": None,
                "spark_logs": None,
                "spark_script": None,
                "improved_script": None,
                "retried": False,
            }
        )
    except Exception as e:
        print("Agent execution error:", e)


def agent_loop():
    while True:
        run_monitor()
        time.sleep(POLL_INTERVAL_SECONDS)


# =========================================================
# GRADIO UI
# =========================================================

def ui_refresh():
    try:
        result = CDE_MANAGER.listJobRuns()
        if result == -1 or not result:
            return "ERROR", "", "ERROR fetching jobs"

        runs = json.loads(result).get("runs", [])
        if not runs:
            return "NO RUNS", "", ""

        latest = max(runs, key=lambda r: r.get("started", ""))
    except Exception as e:
        return f"ERROR: {str(e)}", "", ""

    status = latest.get("status", "UNKNOWN").upper()
    run_id = str(latest.get("id"))

    script = CDE_MANAGER.downloadFileFromResource(
        RESOURCE_NAME,
        APPLICATION_FILE_NAME,
    ) or ""

    logs = CDE_MANAGER.downloadJobRunLogs(run_id, "driver/stdout")
    logs = logs if logs else "No driver stdout logs available"

    return status, script, logs


# =========================================================
# STARTUP
# =========================================================

init_cde()

threading.Thread(target=agent_loop, daemon=True).start()

with gr.Blocks(title="CDE Spark Job Monitor & Auto-Remediator") as demo:
    gr.Markdown("## CDE Spark Job Monitor & Auto-Remediator")

    status_box = gr.Textbox(label="Latest Job Status")
    script_box = gr.Code(label="Spark Script", language="python")

    logs_box = gr.Textbox(
        label="Driver Stdout Logs",
        lines=20,
        interactive=False,
    )

    refresh_btn = gr.Button("Refresh Now")
    refresh_btn.click(
        fn=ui_refresh,
        outputs=[status_box, script_box, logs_box],
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT")),
    )
