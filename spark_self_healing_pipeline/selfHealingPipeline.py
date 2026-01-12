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

import json
import time
import threading
import gradio as gr
from typing import TypedDict, Annotated

from cdepy import (
    cdeconnection,
    cdemanager,
    cdejob,
    utils,
)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os

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
    messages: Annotated[list[BaseMessage], add_messages]

# =========================================================
# TOOLS
# =========================================================

@tool
def connect_to_cde(jobs_api_url: str, user: str, password: str) -> str:
    global CDE_CONNECTION, CDE_MANAGER
    CDE_CONNECTION = cdeconnection.CdeConnection(jobs_api_url, user, password)
    CDE_CONNECTION.setToken()
    CDE_MANAGER = cdemanager.CdeClusterManager(CDE_CONNECTION)
    return "Connected to CDE."


@tool
def validate_job_runs() -> dict:
    return json.loads(CDE_MANAGER.listJobRuns())


@tool
def download_spark_event_logs(job_run_id: str) -> str:
    logs = CDE_MANAGER.downloadJobRunLogs(job_run_id, "driver/event")
    return json.dumps(utils.sparkEventLogParser(logs), indent=2)


@tool
def download_spark_script(resource_name: str, file_name: str) -> str:
    return CDE_MANAGER.downloadFileFromResource(resource_name, file_name)


@tool
def create_new_spark_job(job_name: str, resource_name: str, application_file_name: str) -> str:
    spark_job = cdejob.CdeSparkJob(CDE_CONNECTION)
    job_def = spark_job.createJobDefinition(
        job_name,
        resource_name,
        application_file_name,
        executorMemory="2g",
        executorCores=2,
    )
    CDE_MANAGER.createJob(job_def)
    return f"Created job '{job_name}'."


@tool
def run_cde_job(job_name: str) -> str:
    CDE_MANAGER.runJob(job_name)
    return f"Submitted job '{job_name}'."

# =========================================================
# TOOL REGISTRATION
# =========================================================

TOOLS = [
    connect_to_cde,
    validate_job_runs,
    download_spark_event_logs,
    download_spark_script,
    create_new_spark_job,
    run_cde_job,
]

# =========================================================
# LANGGRAPH SETUP
# =========================================================

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0,
).bind_tools(TOOLS)

def agent(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}

def route(state: AgentState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(TOOLS))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a Spark-on-CDE reliability agent.

Each execution:
1. Check the most recent run of the Spark job.
2. If RUNNING or SUCCEEDED: do nothing.
3. If FAILED:
   - Download Spark event logs
   - Download the Spark script from the CDE resource
   - Identify the root cause
   - Generate a corrected Spark script
   - Create a NEW Spark job
   - Run the new Spark job

Do not retry more than once.
"""

# =========================================================
# CORE EXECUTION
# =========================================================

def run_monitor():
    try:
        app.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Monitor job '{JOB_NAME}'. "
                            f"Resource '{RESOURCE_NAME}', "
                            f"Script '{APPLICATION_FILE_NAME}'."
                        )
                    ),
                ]
            }
        )
    except Exception as e:
        print("Agent execution error:", e)

# =========================================================
# BACKGROUND SCHEDULER (30s)
# =========================================================

def agent_loop():
    while True:
        run_monitor()
        time.sleep(POLL_INTERVAL_SECONDS)

# =========================================================
# INITIALIZATION
# =========================================================

connect_to_cde(JOBS_API_URL, WORKLOAD_USER, WORKLOAD_PASSWORD)

threading.Thread(target=agent_loop, daemon=True).start()

# =========================================================
# GRADIO UI
# =========================================================

def ui_refresh():
    job_runs = json.loads(CDE_MANAGER.listJobRuns())
    latest = job_runs[-1] if job_runs else {}

    status = latest.get("status", "UNKNOWN")
    run_id = latest.get("id", "")

    script = ""
    logs = ""

    if run_id:
        script = download_spark_script(RESOURCE_NAME, APPLICATION_FILE_NAME)

    if status == "FAILED" and run_id:
        logs = download_spark_event_logs(run_id)

    return status, script, logs


with gr.Blocks(title="CDE Spark Job Monitor") as demo:
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
