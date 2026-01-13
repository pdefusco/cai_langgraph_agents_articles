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
#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#***************************************************************************/

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
    utils,
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
# LLM (only for script analysis/fix)
# =========================================================

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0,
)

# =========================================================
# HELPER FUNCTIONS / GRAPH NODES
# =========================================================

def fetch_latest_run(state: AgentState):
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
    if state["latest_run_status"] == "FAILED" and not state["retried"]:
        return "download_artifacts"
    return END

def download_artifacts(state: AgentState):
    run_id = state["latest_run_id"]
    if not run_id:
        raise RuntimeError("No job run ID available for download_artifacts")

    logs_raw = CDE_MANAGER.downloadJobRunLogs(run_id, "driver/event")
    state["spark_logs"] = json.dumps(utils.sparkEventLogParser(logs_raw), indent=2)

    state["spark_script"] = CDE_MANAGER.downloadFileFromResource(
        RESOURCE_NAME, APPLICATION_FILE_NAME
    )
    return state

def llm_fix_script(state: AgentState):
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
            content=(
                f"SPARK SCRIPT:\n{state['spark_script']}\n\n"
                f"SPARK EVENT LOGS:\n{state['spark_logs']}"
            )
        ),
    ]
    response = llm.invoke(prompt)
    state["improved_script"] = response.content
    state["retried"] = True
    return state

def deploy_and_run_fixed_job(state: AgentState):
    new_resource = f"{RESOURCE_NAME}_fixed"
    new_job_name = f"{JOB_NAME}_fixed"

    CDE_MANAGER.createResource(new_resource)
    CDE_MANAGER.uploadFileToResource(
        new_resource, APPLICATION_FILE_NAME, state["improved_script"]
    )

    spark_job = cdejob.CdeSparkJob(CDE_CONNECTION)
    job_def = spark_job.createJobDefinition(
        new_job_name, new_resource, APPLICATION_FILE_NAME, executorMemory="2g", executorCores=2
    )

    CDE_MANAGER.createJob(job_def)
    CDE_MANAGER.runJob(new_job_name)
    return state

def init_cde():
    global CDE_CONNECTION, CDE_MANAGER
    CDE_CONNECTION = cdeconnection.CdeConnection(JOBS_API_URL, WORKLOAD_USER, WORKLOAD_PASSWORD)
    CDE_CONNECTION.setToken()
    CDE_MANAGER = cdemanager.CdeClusterManager(CDE_CONNECTION)

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
# MONITOR LOOP
# =========================================================

def run_monitor():
    try:
        app.invoke({
            "latest_run_id": None,
            "latest_run_status": None,
            "spark_logs": None,
            "spark_script": None,
            "improved_script": None,
            "retried": False,
        })
    except Exception as e:
        print("Execution error:", e)

def agent_loop():
    while True:
        run_monitor()
        time.sleep(POLL_INTERVAL_SECONDS)

# =========================================================
# UTILS FOR GRADIO
# =========================================================

def download_spark_script(resource_name: str, file_name: str) -> str:
    return CDE_MANAGER.downloadFileFromResource(resource_name, file_name)

def download_spark_event_logs(job_run_id: str) -> str:
    logs_raw = CDE_MANAGER.downloadJobRunLogs(job_run_id, "driver/event")
    return json.dumps(utils.sparkEventLogParser(logs_raw), indent=2)

def ui_refresh():
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
        script = download_spark_script(RESOURCE_NAME, APPLICATION_FILE_NAME)
    if status == "FAILED" and run_id:
        logs = download_spark_event_logs(run_id)
    return status, script, logs

# =========================================================
# INITIALIZATION
# =========================================================

init_cde()
threading.Thread(target=agent_loop, daemon=True).start()

# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(title="CDE Spark Job Monitor & Auto-Remediator") as demo:
    gr.Markdown("## CDE Spark Job Monitor & Auto-Remediator")

    status_box = gr.Textbox(label="Latest Job Status")
    script_box = gr.Code(label="Spark Script", language="python")
    logs_box = gr.Code(label="Spark Event Logs", language="json")

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
