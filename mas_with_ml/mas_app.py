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

from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import subprocess
from typing import TypedDict, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import random, os, json, time
from pyspark.sql import SparkSession
import httpx
from urllib.parse import urlparse, urlunparse

LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
LLM_ENDPOINT_BASE_URL = os.environ["LLM_ENDPOINT_BASE_URL"]
LLM_CDP_TOKEN = os.environ["LLM_CDP_TOKEN"]

CLF_MODEL_ID = os.environ["CLF_MODEL_ID"]
CLF_ENDPOINT_BASE_URL = os.environ["CLF_ENDPOINT_BASE_URL"]
CLF_CDP_TOKEN = os.environ["CLF_CDP_TOKEN"]

# ----------------------------
# State Definition
# ----------------------------

class GraphState(TypedDict):
    user_input: str
    extracted_features: Optional[dict]
    classifier_result: Optional[float]
    final_response: Optional[str]


# ----------------------------
# LLM Feature Extraction Agent
# ----------------------------

class Features(BaseModel):
    age: float = Field(default=0)
    credit_card_balance: float = Field(default=0)
    bank_account_balance: float = Field(default=0)
    mortgage_balance: float = Field(default=0)
    sec_bank_account_balance: float = Field(default=0)
    savings_account_balance: float = Field(default=0)
    sec_savings_account_balance: float = Field(default=0)
    total_est_nworth: float = Field(default=0)
    primary_loan_balance: float = Field(default=0)
    secondary_loan_balance: float = Field(default=0)
    uni_loan_balance: float = Field(default=0)
    longitude: float = Field(default=0)
    latitude: float = Field(default=0)
    transaction_amount: float = Field(default=0)


llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0.2,
)

feature_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a feature extraction engine.

        Extract the following numeric features from the user message:

        - age
        - credit_card_balance
        - bank_account_balance
        - mortgage_balance
        - sec_bank_account_balance
        - savings_account_balance
        - sec_savings_account_balance
        - total_est_nworth
        - primary_loan_balance
        - secondary_loan_balance
        - uni_loan_balance
        - longitude
        - latitude
        - transaction_amount

        IMPORTANT:
        - If a value is NOT mentioned, return 0.
        - Only return numeric values.
        - Do NOT infer values unless explicitly stated.
        """
    ),
    ("human", "{input}")
])

feature_chain = feature_prompt | llm.with_structured_output(Features)

report_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a financial risk analysis assistant.

        Generate a concise transaction report based on the structured attributes provided.

        The report should include:
        - Customer profile summary
        - Transaction details
        - Geographic indicators (if available)
        - Overall fraud risk probability
        - Brief risk interpretation

        Keep it professional and under 200 words.
        """
            ),
            (
                "human",
                """
        Transaction Features:
        {features}

        Fraud Probability: {probability}
        """
    )
])

report_chain = report_prompt | llm

marketing_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a banking marketing copywriter.

        Generate a realistic promotional credit card offer email.

        Rules:
        - Use professional but engaging tone.
        - Tailor the offer based on the customer's risk tier.
        - LOW risk → premium rewards, travel perks, higher credit limit.
        - MEDIUM risk → balanced cashback offer.
        - HIGH risk → credit-building card with monitoring tools.
        - Keep under 250 words.
        - Output a full email including subject line.
        """
            ),
            (
                "human",
                """
        Customer Profile:
        - Full Name: {full_name}
        - Email: {email}
        - City: {city}
        - State: {state}
        - Company: {company}
        - Job Title: {job_title}
        - Risk Tier: {risk_tier}

        Recent Fraud Score: {probability}

        Generate the email.
        """
    )
])

marketing_chain = marketing_prompt | llm

FEATURE_ORDER = [
    "age",
    "credit_card_balance",
    "bank_account_balance",
    "mortgage_balance",
    "sec_bank_account_balance",
    "savings_account_balance",
    "sec_savings_account_balance",
    "total_est_nworth",
    "primary_loan_balance",
    "secondary_loan_balance",
    "uni_loan_balance",
    "longitude",
    "latitude",
    "transaction_amount"
]

def feature_extraction_node(state: GraphState):
    result = feature_chain.invoke({"input": state["user_input"]})

    features_dict = result.dict()

    # Hard guarantee no missing values
    for field in Features.model_fields.keys():
        if features_dict.get(field) is None:
            features_dict[field] = 0

    return {
        "extracted_features": features_dict
    }


# ----------------------------
# Classifier Agent (Cloudera AI)
# ----------------------------

def classifier_node(state: GraphState):
    features = state["extracted_features"]

    headers = {
	'Authorization': 'Bearer ' + CLF_CDP_TOKEN,
	'Content-Type': 'application/json'
    }

    httpx_client = httpx.Client(headers=headers)
    client = OpenInferenceClient(base_url=CLF_ENDPOINT_BASE_URL, httpx_client=httpx_client)

    # Check that the server is live, and it has the model loaded
    client.check_server_readiness()
    metadata = client.read_model_metadata(CLF_MODEL_ID)
    metadata_str = json.dumps(json.loads(metadata.json()), indent=2)
    print(metadata_str)

    import time

    ordered_values = [float(features[f]) for f in FEATURE_ORDER]
    print("ordered values sanity check:")
    print(ordered_values)

    payload = {
        "parameters": {
            "content_type": "pd"
        },
        "inputs": [
            {
                "name": "input",
                "datatype": "FP32",
                "shape": [1, 14],
                "data": [ordered_values]
            }
        ]
    }
    start = time.time()
    pred = client.model_infer(
        CLF_MODEL_ID,
        request=InferenceRequest(
            inputs=payload["inputs"]
        ),
    )

    end = time.time()

    resp_json = json.loads(pred.json())
    print(resp_json)
    print(f"latency={end-start}")

    # assuming response like: {"prediction": 1}
    fraudProba = resp_json["outputs"][1]["data"][1]
    print(f"There is a {fraudProba} probability that the credit card transaction is fraudulent")

    return {
        "classifier_result": fraudProba
    }


# ----------------------------
# Action Nodes
# ----------------------------

def action_positive_node(state: GraphState):

    import random

    features = state["extracted_features"]
    probability = state["classifier_result"]

    # -------------------------------------------
    # 1️⃣ Assign Random Demo Customer ID
    # -------------------------------------------
    random_customer_id = random.randint(1, 9999)
    print(f"\nAssigned Demo Customer ID: {random_customer_id}\n")

    # -------------------------------------------
    # 2️⃣ Query Spark for PII
    # -------------------------------------------
    query = f"""
        SELECT *
        FROM default.customers
        WHERE customer_id = {random_customer_id}
        LIMIT 1
    """

    customer_df = spark.sql(query)
    rows = customer_df.collect()

    if not rows:
        return {"final_response": "Customer not found."}

    customer = rows[0].asDict()

    # -------------------------------------------
    # 3️⃣ Call LLM to Generate Offer
    # -------------------------------------------
    response = marketing_chain.invoke({
        "full_name": customer["full_name"],
        "email": customer["email"],
        "city": customer["city"],
        "state": customer["state"],
        "company": customer["company"],
        "job_title": customer["job_title"],
        "risk_tier": customer["risk_tier"],
        "probability": f"{probability:.2%}"
    })

    email_text = response.content

    print("\n===== GENERATED MARKETING EMAIL =====\n")
    print(email_text)
    print("\n=====================================\n")

    return {
        "final_response": email_text
    }


def action_negative_node(state: GraphState):
    features = state["extracted_features"]
    probability = state["classifier_result"]

    transaction_amount = features.get("transaction_amount", 0)
    latitude = features.get("latitude", 0)
    longitude = features.get("longitude", 0)

    email_message = f"""
    ===== FRAUD ALERT NOTIFICATION =====

    To: customer@email.com
    Subject: Urgent: Suspicious Credit Card Transaction Detected

    Dear Customer,

    We detected a potentially fraudulent credit card transaction.

    Transaction Details:
    - Amount: ${transaction_amount}
    - Location (lat/long): ({latitude}, {longitude})
    - Estimated Fraud Probability: {probability:.2%}

    For your protection, this transaction has been temporarily flagged.

    If this was you, please confirm via your banking app.
    If not, contact our fraud department immediately.

    Sincerely,
    Fraud Prevention Team

    =====================================
    """

    print(email_message)

    return {
        "final_response": "Fraud alert email notification generated."
    }


# ----------------------------
# Router Logic
# ----------------------------

def router(state: GraphState):
    probability = float(state["classifier_result"])

    print(f"Routing decision — fraud probability: {probability}")

    if probability < 0.5:
        print("→ Routing to action_positive")
        return "action_positive"
    else:
        print("→ Routing to action_negative")
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
# Streamlit App
# ----------------------------
import streamlit as st

def run_streamlit_app(graph):
    st.set_page_config(page_title="Multi-Agent System with Machine Learning", layout="wide")
    st.title("Multi-Agent System with Machine Learning")

    # 1️⃣ User Input
    user_message = st.text_area(
        "Enter user message / transaction description:",
        value="Hi, I am a 45 year old male interested in financial planning.",
        height=150
    )

    if st.button("Run Classifier"):

        with st.spinner("Processing transaction..."):
            result = graph.invoke({"user_input": user_message})

        # 2️⃣ Extract outputs
        fraud_probability = result.get("classifier_result", None)
        final_response = result.get("final_response", "")
        extracted_features = result.get("extracted_features", {})

        # Determine which action was chosen based on probability
        if fraud_probability is not None:
            if fraud_probability < 0.5:
                action_chosen = "Positive (Marketing Offer)"
                action_color = "green"
            else:
                action_chosen = "Negative (Fraud Alert)"
                action_color = "red"

            # 3️⃣ Display Classifier Info
            st.subheader("Classifier Output")
            col1, col2 = st.columns([1, 3])

            with col1:
                st.metric("Fraud Probability", f"{fraud_probability:.2%}")
                st.progress(fraud_probability)

            with col2:
                st.markdown(
                    f"<div style='padding:10px; border-radius:5px; background-color:{action_color}; color:white; font-weight:bold; text-align:center'>{action_chosen}</div>",
                    unsafe_allow_html=True
                )

            # 4️⃣ Display Action Output
            st.subheader("Action Output")
            st.markdown(final_response.replace("\n", "  \n"))

            # 5️⃣ Display Extracted Features
            st.subheader("Extracted Features")
            st.json(extracted_features)


# ----------------------------
# Main CLI-style Example
# ----------------------------
if __name__ == "__main__":
    run_streamlit_app(graph)
