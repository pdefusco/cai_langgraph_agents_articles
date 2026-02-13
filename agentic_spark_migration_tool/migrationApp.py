import os
import shutil
from typing import Dict, Any, List

import gradio as gr
from pydantic import BaseModel, ValidationError

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

import chromadb
from chromadb.config import Settings
from cdepy import cdejob, cderesource, cdemanager
from cdepy.connection import CdeConnection

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SCRIPTS_DIR = "scripts"
os.makedirs(SCRIPTS_DIR, exist_ok=True)

JOBS_API_URL = os.environ["JOBS_API_URL"]
WORKLOAD_USER = os.environ["WORKLOAD_USER"]
WORKLOAD_PASSWORD = os.environ["WORKLOAD_PASSWORD"]

JOB_NAME = os.environ["JOB_NAME"]
RESOURCE_NAME = os.environ["RESOURCE_NAME"]
APPLICATION_FILE_NAME = os.environ["APPLICATION_FILE_NAME"]

LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
LLM_ENDPOINT_BASE_URL = os.environ["LLM_ENDPOINT_BASE_URL"]
LLM_CDP_TOKEN = os.environ["LLM_CDP_TOKEN"]

CHROMA_DIR = "/home/cdsw/chroma"
COLLECTION_NAME = "spark_submit_cde_mappings"

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

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0.2,
)

# -------------------------------------------------------------------
# Chroma (existing collection)
# -------------------------------------------------------------------

chroma_client = chromadb.Client(
    Settings(persist_directory=CHROMA_DIR)
)

spark_submit_collection = chroma_client.get_collection(
    COLLECTION_NAME
)

# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------

class ParsedSparkSubmit(BaseModel):
    executor_memory: str | None = None
    executor_cores: int | None = None
    num_executors: int | None = None
    spark_conf: Dict[str, str] = {}
    args: List[str] = []


class CdeFilesResourceSpec(BaseModel):
    resource_name: str
    local_files: List[str]


class CdeSparkJobSpec(BaseModel):
    job_name: str
    resource_name: str
    application_file: str
    executorMemory: str | None = None
    executorCores: int | None = None
    numExecutors: int | None = None
    spark_conf: Dict[str, str] = {}
    args: List[str] = []


class AgentState(BaseModel):
    spark_submit: str
    script_name: str
    local_script_path: str

    job_name: str
    resource_name: str

    parsed_submit: ParsedSparkSubmit | None = None
    rag_examples: List[str] = []

    resource_spec: CdeFilesResourceSpec | None = None
    job_spec: CdeSparkJobSpec | None = None

    errors: List[str] = []


# -------------------------------------------------------------------
# LangGraph Nodes
# -------------------------------------------------------------------

def parse_spark_submit(state: AgentState) -> AgentState:
    prompt = f"""
Extract parameters from this spark-submit command.

Return JSON with:
- executor_memory
- executor_cores
- num_executors
- spark_conf
- args

Do NOT infer application file.

Spark submit:
{state.spark_submit}
"""
    response = llm.invoke(prompt).content
    state.parsed_submit = ParsedSparkSubmit.model_validate_json(response)
    return state


def retrieve_examples(state: AgentState) -> AgentState:
    results = spark_submit_collection.query(
        query_texts=[state.spark_submit],
        n_results=3
    )

    state.rag_examples = results["documents"][0]
    return state


def generate_cde_specs(state: AgentState) -> AgentState:
    prompt = f"""
You are a Python coding agent tasked with creating Cloudera Data Engineering (CDE) Spark jobs
using the cdepy library. You will be given retrieved documents from a vector database.
These documents are canonical examples of spark-submit commands mapped to CDE job resources,
job definitions, and cdepy execution sequences.

Your job is to:

1. Follow the retrieved examples exactly for structure and ordering.
2. Ensure all resource → upload → job creation steps are preserved.
3. Map --conf flags from the spark-submit input only to the sparkConf dictionary in cdepy.
4. Use cdepy.CdeFilesResource and cdepy.CdeSparkJob methods as in the examples; do not invent new methods.
5. Include a TODO comment if any required information is missing from the input.
6. Keep the output executable and deterministic-friendly; do not hallucinate filenames, resources, or configs.

The job_name and resource_name are already provided.
You MUST use them exactly as given. Do NOT modify them.

Retrieved examples:
{state.rag_examples}

Parsed spark-submit:
{state.parsed_submit.model_dump()}

Uploaded PySpark script:
{state.script_name}

Return JSON only in the following format:
{
  "cde_files_resource": {
    "resource_name": "{state.resource_name}",
    "local_files": ["{state.script_name}"]
  },
  "cde_spark_job": {
    "job_name": "{state.job_name}",
    "resource_name": "{state.resource_name}",
    "application_file": "{state.script_name}",
    "executorMemory": "...",
    "executorCores": ...,
    "numExecutors": ...,
    "spark_conf": {},
    "args": []
  }
}

"""
    response = llm.invoke(prompt).content
    data = eval(response)  # trusted environment assumption

    state.resource_spec = CdeFilesResourceSpec(
        **data["cde_files_resource"]
    )
    state.job_spec = CdeSparkJobSpec(
        **data["cde_spark_job"]
    )
    return state


def validate_specs(state: AgentState) -> AgentState:
    errors = []

    if state.script_name not in state.resource_spec.local_files:
        errors.append("Uploaded script missing from CDE resource")

    if state.job_spec.application_file != state.script_name:
        errors.append("Application file mismatch")

    if errors:
        state.errors = errors

    return state


def execute_cde(state: AgentState) -> AgentState:
    manager = cdemanager.CdeClusterManager(myCdeConnection)

    # Create resource
    resource = cderesource.CdeFilesResource(
        state.resource_spec.resource_name
    )
    resource_def = resource.createResourceDefinition()
    manager.createResource(resource_def)

    # Upload script
    manager.uploadFileToResource(
        state.resource_spec.resource_name,
        SCRIPTS_DIR,
        state.script_name
    )

    # Create job
    spark_job = cdejob.CdeSparkJob(myCdeConnection)
    job_def = spark_job.createJobDefinition(
        state.job_spec.job_name,
        state.job_spec.resource_name,
        state.job_spec.application_file,
        executorMemory=state.job_spec.executorMemory,
        executorCores=state.job_spec.executorCores,
        numExecutors=state.job_spec.numExecutors
    )

    manager.createJob(job_def)
    return state

# -------------------------------------------------------------------
# LangGraph wiring
# -------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("parse_submit", parse_spark_submit)
graph.add_node("retrieve_examples", retrieve_examples)
graph.add_node("generate_specs", generate_cde_specs)
graph.add_node("validate", validate_specs)
graph.add_node("execute", execute_cde)

graph.set_entry_point("parse_submit")

graph.add_edge("parse_submit", "retrieve_examples")
graph.add_edge("retrieve_examples", "generate_specs")
graph.add_edge("generate_specs", "validate")

graph.add_conditional_edges(
    "validate",
    lambda s: "execute" if not s.errors else END
)

graph.add_edge("execute", END)

agent = graph.compile()

# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------

def run_agent(spark_submit_text, pyspark_file):
    script_path = os.path.join(SCRIPTS_DIR, pyspark_file.name)
    shutil.copy(pyspark_file.name, script_path)

    state = AgentState(
        spark_submit=spark_submit_text,
        script_name=pyspark_file.name,
        local_script_path=script_path,
        job_name=JOB_NAME,
        resource_name=RESOURCE_NAME
    )

    final_state = agent.invoke(state)

    if final_state.errors:
        return {
            "status": "error",
            "errors": final_state.errors
        }

    return {
        "status": "success",
        "resource": final_state.resource_spec.model_dump(),
        "job": final_state.job_spec.model_dump()
    }


with gr.Blocks() as demo:
    gr.Markdown("## Create CDE Spark Job (LangGraph + RAG)")

    spark_submit = gr.Textbox(
        label="Spark Submit Command",
        lines=6
    )

    pyspark_file = gr.File(
        label="Upload PySpark Script (.py)",
        file_types=[".py"]
    )

    output = gr.JSON(label="Result")

    btn = gr.Button("Create CDE Job")
    btn.click(run_agent, [spark_submit, pyspark_file], output)

demo.launch()
