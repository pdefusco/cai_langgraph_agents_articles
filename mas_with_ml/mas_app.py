#****************************************************************************
# (C) Cloudera, Inc. 2019-2026
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

import subprocess
import json
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


# ----------------------------
# State Definition
# ----------------------------

class GraphState(TypedDict):
    user_input: str
    extracted_features: Optional[dict]
    classifier_result: Optional[int]
    final_response: Optional[str]


# ----------------------------
# LLM Feature Extraction Agent
# ----------------------------

class Features(BaseModel):
    age: int
    gender: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

feature_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured features (age and gender) from the user message."),
    ("human", "{input}")
])

feature_chain = feature_prompt | llm.with_structured_output(Features)


def feature_extraction_node(state: GraphState):
    result = feature_chain.invoke({"input": state["user_input"]})
    return {
        "extracted_features": result.dict()
    }


# ----------------------------
# Classifier Agent (Cloudera AI)
# ----------------------------

CLOUDERA_ENDPOINT = "https://your-cloudera-endpoint/model/predict"
CLOUDERA_TOKEN = "your_token"


def classifier_node(state: GraphState):
    features = state["extracted_features"]

    payload = json.dumps(features)

    curl_command = [
        "curl",
        "-X", "POST",
        CLOUDERA_ENDPOINT,
        "-H", f"Authorization: Bearer {CLOUDERA_TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", payload
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)

    response_json = json.loads(result.stdout)

    # assuming response like: {"prediction": 1}
    prediction = response_json["prediction"]

    return {
        "classifier_result": prediction
    }


# ----------------------------
# Action Nodes
# ----------------------------

def action_positive_node(state: GraphState):
    return {
        "final_response": "Classifier returned 1 → Taking Action A (e.g., offer premium service)."
    }


def action_negative_node(state: GraphState):
    return {
        "final_response": "Classifier returned 0 → Taking Action B (e.g., standard workflow)."
    }


# ----------------------------
# Router Logic
# ----------------------------

def router(state: GraphState):
    if state["classifier_result"] == 1:
        return "action_positive"
    else:
        return "action_negative"


# ----------------------------
# Build LangGraph
# ----------------------------

builder = StateGraph(GraphState)

builder.add_node("feature_extraction", feature_extraction_node)
builder.add_node("classifier", classifier_node)
builder.add_node("action_positive", action_positive_node)
builder.add_node("action_negative", action_negative_node)

builder.set_entry_point("feature_extraction")

builder.add_edge("feature_extraction", "classifier")

builder.add_conditional_edges(
    "classifier",
    router,
    {
        "action_positive": "action_positive",
        "action_negative": "action_negative"
    }
)

builder.add_edge("action_positive", END)
builder.add_edge("action_negative", END)

graph = builder.compile()


# ----------------------------
# Run Example
# ----------------------------

if __name__ == "__main__":
    user_message = "Hi, I am a 45 year old male interested in financial planning."

    result = graph.invoke({
        "user_input": user_message
    })

    print(result["final_response"])
