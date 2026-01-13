#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#***************************************************************************/

import json
import time
import os
from typing import TypedDict

import gradio as gr

from cdepy import cdeconnection, cdemanager, cdejob, utils
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
# GLOBAL CDE OBJECTS (SESSION PERSISTENCE)
# =========================================================

CDE_CONNECTION = None
CDE_MANAGER = None

def init_cde():
    """Initialize CDE connection and manager once at startup."""
    global CDE_CONNECTION, CDE_MANAGER
    if CDE_CONNECTION is None or CDE_MANAGER is None:
        CDE_CONNECTION = cdeconnection.CdeConnection(
            JOBS_API_URL, WORKLOAD_USER, WORKLOAD_PASSWORD
        )
        CDE_CONNECTION.setToken()  # token set once
        CDE_MANAGER = cdemanager.CdeClusterManager(CDE_CONNECTION)
        print("CDE initialized and token set successfully")

init_cde()  # call at startup

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
# LLM (USED ONLY FOR SCRIPT ANALYSIS + FIX)
# =========================================================

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0,
)

# =========================================================
# TOOL FUNCTIONS / LANGGRAPH NODES
# =========================================================

def fetch_latest_run(state: AgentState):
    """Fetch the latest Spark job run deterministically."""
    result = CDE_MANAGER.listJobRuns()
    if result == -1 or not result:
        raise RuntimeError("Failed to list CDE job runs (API returned -1)")

    try:
        runs = json.loads(result)
    except Exception as e:
        raise RuntimeError(f"Invalid response from listJobRuns(): {result}") from e

    if not runs:
        state["latest_run_id"] = None
        state["latest_run_status"] = None
    else:
        latest = runs[-1]
        state["latest_run_id"] = latest.get("id")
        state["latest_run_status"] = latest.get("status")
    return state

def route_on_status(state: AgentState):
    """Conditional routing: only download artifacts if last run FAILED and not retried."""
    if state["latest_run_status"] == "FAILED" and not state["retried"]:
        return "download_artifacts"
    return END

def download_artifacts(state: AgentState):
    """Download Spark logs and script for the failed job."""
    run_id = state["latest_run_id"]
    logs_raw = CDE_MANAGER.downloadJobRunLogs(run_id, "driver/event")
    state["spark_logs"] = json.dumps(utils.sparkEventLogParser(logs_raw), indent=2)
    state["spark_script"] = CDE_MANAGER.downloadFileFromResource(RESOURCE_NAME, APPLICATION_FILE_NAME)
    return state

def llm_fix_script(state: AgentState):
    """Use LLM to analyze logs and fix Spark script."""
    prompt = [
        SystemMessage(
            content=(
                "You are a senior Spark engineer.\n"
                "Analyze the Spark event logs and fix the application.\n"
                "Preserve original intent.\n"
                "Return ONLY valid Python Spark code."
            )
        ),
        HumanMessage(
            content=f"SPARK SCRIPT:\n{state['spark_script']}\n\nSPARK EVENT LOGS:\n{state['spark_logs']}"
        ),
    ]
    response = llm.invoke(prompt)
    state["improved_script"] = response.content
    state["retried"] = True
    return state

def deploy_and_run_fixed_job(state: AgentState):
    """Deterministically deploy fixed script and run new Spark job."""
    new_resource = f"{RESOURCE_NAME}_fixed"
    new_job_name = f"{JOB_NAME}_fixed"

    CDE_MANAGER.createResource(new_resource)
    CDE_MANAGER.uploadFileToResource(new_resource, APPLICATION_FILE_NAME, state["improved_script"])

    spark_job = cdejob.CdeSparkJob(CDE_CONNECTION)
    job_def = spark_job.createJobDefinition(new_job_name, new_resource, APPLICATION_FILE_NAME,
                                            executorMemory="2g", executorCores=2)
    CDE_MANAGER.createJob(job_def)
    CDE_MANAGER.runJob(new_job_name)
    return state

# =========================================================
# LANGGRAPH SETUP
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
# GRADIO UI
# =========================================================

def ui_refresh():
    """Deterministic UI refresh with session-persistent CDE connection."""
    try:
        result = CDE_MANAGER.listJobRuns()
        if result == -1 or not result:
            return "ERROR fetching jobs", "", ""

        job_runs = json.loads(result)
        latest = job_runs[-1] if job_runs else {}
    except Exception as e:
        return f"ERROR: {str(e)}", "", ""

    status = latest.get("status", "UNKNOWN")
    run_id = latest.get("id", "")

    script = ""
    logs = ""

    if run_id:
        script = CDE_MANAGER.downloadFileFromResource(RESOURCE_NAME, APPLICATION_FILE_NAME)

    if status == "FAILED" and run_id:
        logs_raw = CDE_MANAGER.downloadJobRunLogs(run_id, "driver/event")
        logs = json.dumps(utils.sparkEventLogParser(logs_raw), indent=2)

    return status, script, logs

with gr.Blocks(title="CDE Spark Job Monitor & Auto-Remediator") as demo:
    gr.Markdown("## CDE Spark Job Monitor & Auto-Remediator")

    status_box = gr.Textbox(label="Latest Job Status")
    script_box = gr.Code(label="Spark Script", language="python")
    logs_box = gr.Code(label="Spark Event Logs", language="json")

    refresh_btn = gr.Button("Refresh Now")
    refresh_btn.click(fn=ui_refresh, outputs=[status_box, script_box, logs_box])

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT", 8080)),
    )
