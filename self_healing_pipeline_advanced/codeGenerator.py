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
# LANGGRAPH NODES
# =========================================================

def llm_generate_scripts(state: AgentState) -> AgentState:
    """
    Ask the LLM to generate 5 non-trivial PySpark scripts that intentionally fail.
    """
    prompt = [
        SystemMessage(
            content=(
                "You are a senior Spark engineer.\n\n"
                "Generate EXACTLY 5 PySpark applications.\n\n"
                "Rules:\n"
                "- Each script must be a complete Spark application\n"
                "- Each must contain real (non-trivial) logic\n"
                "- Each must FAIL intentionally\n\n"
                "Failure modes (use each once):\n"
                "1. Missing table (AnalysisException)\n"
                "2. Python TypeError\n"
                "3. None dereference\n"
                "4. Divide by zero inside a Spark UDF\n"
                "5. DataFrame schema mismatch during union\n\n"
                "Return STRICT JSON ONLY:\n"
                "{ \"scripts\": [ { \"name\": str, \"description\": str, \"code\": str } ] }\n"
                "No markdown. No backticks."
            )
        )
    ]

    response = llm.invoke(prompt)
    payload = json.loads(response.content)

    state["scripts"] = payload["scripts"]
    return state


def create_resource_once(state: AgentState) -> AgentState:
    """
    Create the CDE Files Resource exactly once.
    """
    if state.get("resource_created"):
        return state

    resource = cderesource.CdeFilesResource(RESOURCE_NAME)
    CDE_MANAGER.createResource(resource.createResourceDefinition())

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
