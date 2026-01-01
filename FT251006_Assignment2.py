import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import tiktoken

# --- API Configuration ---
# If running locally, you can still use st.secrets by creating a file at:
# .streamlit/secrets.toml
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

# --- Initialize Model & Chain ---
# Note: "models/" prefix removed for better compatibility
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.7)

prompt_template_swot = """
You are a management consultant. Analyze the following information and provide a detailed SWOT analysis.
Identify Strengths, Weaknesses, Opportunities, and Threats.

{context}

Format the response EXACTLY as follows:

**Strengths:**
- [Strength 1]
...

**Weaknesses:**
- [Weakness 1]
...

**Opportunities:**
- [Opportunity 1]
...

**Threats:**
- [Threat 1]
...
"""

prompt_swot = PromptTemplate(input_variables=["context"], template=prompt_template_swot)

# Modern LCEL Chain: prompt piped to llm
chain = prompt_swot | llm

# --- Session State ---
if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0

# Token encoder
encoder = tiktoken.get_encoding("cl100k_base")

# --- UI Setup ---
st.set_page_config(page_title="SWOT Analysis Agent")
st.title("SWOT Analysis Agent (Gemini 1.5 Pro)")

text_input = st.text_area("Enter organization information:", height=200)

if st.button("Generate SWOT Analysis"):
    if not text_input:
        st.warning("Please enter some text first.")
    else:
        with st.spinner('Consulting Gemini...'):
            try:
                # Use .invoke for modern LangChain chains
                response = chain.invoke({"context": text_input})
                swot_result = response.content

                st.subheader("SWOT Analysis Result:")
                st.markdown(swot_result)

                # --- Token Calculation ---
                query_tokens = len(encoder.encode(text_input))
                res_tokens = len(encoder.encode(swot_result))
                st.session_state.tokens_consumed += (query_tokens + res_tokens)

                # Sidebar Metrics
                st.sidebar.metric("Tokens This Run", query_tokens + res_tokens)
                st.sidebar.metric("Total Session Tokens", st.session_state.tokens_consumed)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
