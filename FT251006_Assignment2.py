import os
import streamlit as st
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --- API Configuration ---
# Uses Streamlit Secrets (mandatory for Streamlit Cloud in 2026)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add GOOGLE_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- Initialize Model ---
# 'gemini-3-flash-preview' is the highest-quota model for Free Tier in Jan 2026.
# If you still get a 404, fallback to 'gemini-2.5-flash'.
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    google_api_key=api_key, 
    temperature=0.7
)

# --- Define the prompt template ---
prompt_template_swot = """
You are a management consultant. Analyze the provided context and deliver a professional SWOT analysis.

{context}

Format the response with these exact headers:
**Strengths:**
**Weaknesses:**
**Opportunities:**
**Threats:**

Also, include a | SWOT Matrix | as a Markdown table at the end.
"""

prompt_swot = PromptTemplate(input_variables=["context"], template=prompt_template_swot)

# Modern LCEL Chain
swot_chain = prompt_swot | llm

# --- UI Setup ---
st.set_page_config(page_title="SWOT Analysis AI", layout="wide")
st.title("💼 SWOT Analysis Agent (Gemini 3)")

# Initialize session state for tracking
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0

text_input = st.text_area("Enter organization info to analyze:", height=200)

if st.button("Generate SWOT"):
    if text_input:
        with st.spinner("Analyzing with Gemini 3 Flash..."):
            try:
                # Execution
                response = swot_chain.invoke({"context": text_input})
                result = response.content
                
                # UI Output
                st.subheader("Analysis Results")
                st.markdown(result)
                
                # Token Tracking
                encoder = tiktoken.get_encoding("cl100k_base")
                tokens = len(encoder.encode(text_input)) + len(encoder.encode(result))
                st.session_state.total_tokens += tokens
                st.sidebar.metric("Tokens Consumed", st.session_state.total_tokens)
                
            except Exception as e:
                if "429" in str(e):
                    st.error("Free Tier Quota Exceeded. Please wait 60 seconds or switch to a Paid Tier key.")
                elif "404" in str(e):
                    st.error("Model not found. Try updating the model name to 'gemini-2.5-flash'.")
                else:
                    st.error(f"Error: {e}")
    else:
        st.warning("Please provide input text.")
