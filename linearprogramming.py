import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# ========================================================================
# 1. LLM SCHEMA & AI PARSER (GOOGLE GEMINI)
# ========================================================================
class LPProblemSchema(BaseModel):
    sense: str = Field(
        description="Optimization sense: 'Maximize' or 'Minimize'"
    )
    objective_function: str = Field(
        description="Algebraic objective expression without 'Maximize' or 'Minimize' prefix, e.g., '40*Utility + 30*RawMaterial'"
    )
    constraints: List[str] = Field(
        description="List of constraint equations using <=, >=, or =, e.g., ['2*Utility + RawMaterial <= 100', 'Utility + 2*RawMaterial <= 80']"
    )


def parse_lp_with_gemini(user_prompt: str, api_key: str = None) -> LPProblemSchema:
    """Extracts LP parameters from natural language using Google AI Studio Gemini API."""
    resolved_api_key = api_key or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not resolved_api_key:
        raise ValueError("Google API Key not found. Please add GOOGLE_API_KEY to Streamlit Secrets or environment variables.")

    # Using active current-generation model endpoint
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=resolved_api_key
    )

    structured_llm = llm.with_structured_output(LPProblemSchema)

    system_prompt = (
        "You are an expert operations research assistant. Parse the user's natural language linear programming problem. "
        "Extract decision variables, formulate the algebraic objective function, and construct clean constraint equations. "
        "Do NOT include unit labels or currency signs in algebraic terms. Standardize variable names using clean, valid Python identifier names."
    )

    return structured_llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])


# ========================================================================
# 2. STREAMLIT APPLICATION INTERFACE
# ========================================================================
def main():
    st.set_page_config(page_title="LP Problem Assistant", layout="wide")
    st.title("Linear Programming Parser & Solver")

    # API Key Handling in Sidebar
    st.sidebar.header("Configuration")
    user_api_key = st.sidebar.text_input(
        "Google AI Studio API Key", 
        type="password", 
        help="Enter key if not set in Streamlit secrets or environment variables."
    )

    # Input Form
    st.subheader("Natural Language Problem Input")
    user_prompt = st.text_area(
        "Describe your linear programming problem:",
        height=150,
        placeholder="e.g., Maximize profit where profit is 40*x + 30*y, subject to 2*x + y <= 100 and x + 2*y <= 80."
    )

    if st.button("Parse LP Problem"):
        if not user_prompt.strip():
            st.warning("Please enter a problem description first.")
            return

        with st.spinner("Parsing problem with Gemini..."):
            try:
                parsed_lp = parse_lp_with_gemini(user_prompt, api_key=user_api_key)
                
                st.success("Successfully Parsed LP Problem!")
                
                # Display Results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Optimization Sense", parsed_lp.sense)
                    st.text_input("Objective Function", value=parsed_lp.objective_function, disabled=True)
                
                with col2:
                    st.subheader("Constraints")
                    for idx, c in enumerate(parsed_lp.constraints, start=1):
                        st.text(f"Constraint {idx}: {c}")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    main()
