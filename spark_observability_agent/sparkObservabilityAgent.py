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
Spark Metrics Multi-Agent App (LangGraph + Supervisor + Gradio)

Assumptions:
- SparkSession with Hive support is available
- Hive table `spark_app_metrics` already exists
- LLM endpoint will be added later via ChatOpenAI
"""

# =============================
# Imports
# =============================
from typing import TypedDict, List, Dict, Any, Optional
from pyspark.sql import SparkSession
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import gradio as gr
import os

# =============================
# Spark Session
# =============================
spark = SparkSession.builder.config("spark.kerberos.access.hadoopFileSystems","s3a://go01-demo/").getOrCreate()

# =============================
# CAI Environment Variables
# =============================
MODEL_ID = os.environ["MODEL_ID"]
ENDPOINT_BASE_URL = os.environ["ENDPOINT_BASE_URL"]
CDP_TOKEN = os.environ["CDP_TOKEN"]

# =============================
# LLM
# =============================
llm = ChatOpenAI(
    model_name=MODEL_ID,
    openai_api_base=ENDPOINT_BASE_URL,
    openai_api_key=CDP_TOKEN
)

# =============================
# LangGraph State
# =============================
class MetricsState(TypedDict):
    user_question: str

    intent: Optional[str]
    requested_metrics: Optional[List[str]]

    query_plan: Optional[Dict[str, Any]]

    sql_query: Optional[str]
    query_result: Optional[List[Dict[str, Any]]]

    analysis: Optional[str]
    final_answer: Optional[str]

    query_attempts: int
    supervisor_decision: Optional[str]


# =============================
# Spark SQL Executor (Guarded)
# =============================
ALLOWED_TABLE = "spark_app_metrics"

def validate_sql(sql: str):
    sql_lower = sql.lower()
    if any(x in sql_lower for x in ["insert", "update", "delete", "drop"]):
        raise ValueError("Only SELECT queries are allowed")
    if ALLOWED_TABLE not in sql_lower:
        raise ValueError("Query must read from spark_app_metrics")
    if "limit" not in sql_lower:
        raise ValueError("LIMIT clause is required")


def spark_sql_executor(sql: str) -> List[Dict[str, Any]]:
    validate_sql(sql)
    df = spark.sql(sql)
    return df.limit(500).toPandas().to_dict("records")


# =============================
# Agents
# =============================
def intent_agent(state: MetricsState) -> MetricsState:
    q = state["user_question"].lower()
    if "slow" in q or "regression" in q:
        intent = "performance_regression"
        metrics = ["task_duration_ms", "shuffle_read_mb"]
    elif "cost" in q:
        intent = "cost_analysis"
        metrics = ["executor_cpu_time"]
    else:
        intent = "general_observability"
        metrics = ["task_duration_ms"]

    return {**state, "intent": intent, "requested_metrics": metrics}


def metrics_planner_agent(state: MetricsState) -> MetricsState:
    if state["intent"] == "performance_regression":
        plan = {
            "group_by": ["stage_id"],
            "aggregations": {
                "task_duration_ms": "p95",
                "shuffle_read_mb": "avg"
            },
            "time_window_days": 1
        }
    else:
        plan = {
            "group_by": ["app_id"],
            "aggregations": {"task_duration_ms": "avg"},
            "time_window_days": 1
        }

    return {**state, "query_plan": plan}


def text_to_sql_agent(state: MetricsState) -> MetricsState:
    plan = state["query_plan"]
    group_by = ", ".join(plan["group_by"])

    select_exprs = []
    for col, agg in plan["aggregations"].items():
        if agg == "avg":
            select_exprs.append(f"avg({col}) AS avg_{col}")
        elif agg == "p95":
            select_exprs.append(
                f"percentile_approx({col}, 0.95) AS p95_{col}"
            )

    select_sql = ",\n  ".join(plan["group_by"] + select_exprs)

    sql = f"""
    SELECT
      {select_sql}
    FROM spark_app_metrics
    WHERE event_time >= current_timestamp() - INTERVAL {plan['time_window_days']} DAYS
    GROUP BY {group_by}
    LIMIT 100
    """

    return {**state, "sql_query": sql.strip()}


def execute_query_node(state: MetricsState) -> MetricsState:
    result = spark_sql_executor(state["sql_query"])
    return {
        **state,
        "query_result": result,
        "query_attempts": state.get("query_attempts", 0) + 1
    }


MAX_QUERY_ATTEMPTS = 2

def supervisor_agent(state: MetricsState) -> MetricsState:
    attempts = state.get("query_attempts", 0)
    sql = state.get("sql_query")
    result = state.get("query_result")

    if result:
        decision = "proceed_to_analysis"
    elif attempts >= MAX_QUERY_ATTEMPTS:
        decision = "force_finalize"
    elif not sql:
        decision = "generate_sql"
    elif "executor_id" in sql.lower():
        decision = "simplify_plan"
    else:
        decision = "execute_query"

    return {**state, "supervisor_decision": decision}


def simplify_plan_agent(state: MetricsState) -> MetricsState:
    simplified_plan = {
        **state["query_plan"],
        "group_by": ["stage_id"]
    }
    return {**state, "query_plan": simplified_plan, "sql_query": None}


def metrics_analyst_agent(state: MetricsState) -> MetricsState:
    rows = state["query_result"]
    if not rows:
        analysis = "No significant anomalies detected."
    else:
        analysis = (
            f"Analyzed {len(rows)} metric groups. "
            f"Potential skew or regression detected."
        )

    return {**state, "analysis": analysis}


def final_insight_agent(state: MetricsState) -> MetricsState:
    return {
        **state,
        "final_answer": (
            f"Intent: {state['intent']}\n\n"
            f"{state.get('analysis', '')}"
        )
    }


# =============================
# Supervisor Router
# =============================
def supervisor_router(state: MetricsState) -> str:
    decision = state["supervisor_decision"]
    if decision == "generate_sql":
        return "text_to_sql"
    if decision == "execute_query":
        return "execute"
    if decision == "simplify_plan":
        return "simplify"
    if decision == "proceed_to_analysis":
        return "analyze"
    return "final"


# =============================
# Build LangGraph
# =============================
graph = StateGraph(MetricsState)

graph.add_node("intent", intent_agent)
graph.add_node("planner", metrics_planner_agent)
graph.add_node("text_to_sql", text_to_sql_agent)
graph.add_node("execute", execute_query_node)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("simplify", simplify_plan_agent)
graph.add_node("analyze", metrics_analyst_agent)
graph.add_node("final", final_insight_agent)

graph.set_entry_point("intent")
graph.add_edge("intent", "planner")
graph.add_edge("planner", "supervisor")

graph.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "text_to_sql": "text_to_sql",
        "execute": "execute",
        "simplify": "simplify",
        "analyze": "analyze",
        "final": "final",
    }
)

graph.add_edge("text_to_sql", "supervisor")
graph.add_edge("execute", "supervisor")
graph.add_edge("simplify", "supervisor")
graph.add_edge("final", END)

app = graph.compile()


# =============================
# Gradio UI
# =============================
PRESET_QUESTIONS = [
    "Why did my Spark job slow down yesterday?",
    "Did any stages experience performance regressions?",
    "Are there cost anomalies in recent Spark jobs?",
    "Custom question"
]

def run_agent(selected_question, custom_question):
    question = (
        custom_question
        if selected_question == "Custom question"
        else selected_question
    )

    result = app.invoke({
        "user_question": question,
        "query_attempts": 0
    })

    return result["final_answer"]


with gr.Blocks(title="Spark Metrics Agent") as demo:
    gr.Markdown("## 🔍 Spark Metrics Investigation Agent")

    question_radio = gr.Radio(
        PRESET_QUESTIONS,
        label="Select a question"
    )

    custom_input = gr.Textbox(
        label="Custom question",
        placeholder="Ask anything about Spark performance..."
    )

    output = gr.Textbox(
        label="Agent Response",
        lines=8
    )

    run_button = gr.Button("Run Analysis")

    run_button.click(
        fn=run_agent,
        inputs=[question_radio, custom_input],
        outputs=output
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=16).launch(share=False,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
