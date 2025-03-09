#!/usr/bin/env python
# coding: utf-8

# Aditya Gupta, FT251006, Section-1

# Objective: Using Streamlit and LangChain design an AI Agent that takes as input information about an
# organization such as NovaEdge (below) and emits a detailed SWOT analysis. The Analysis should
# include a visual representation of the key points of the SWOT analysis.

# Import necessary libraries
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tiktoken

# Initialize token counters in Streamlit session state
if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0

# Streamlit UI Setup
st.set_page_config(page_title="SWOT Analysis Agent (LangChain + Gemini)")
st.title("SWOT Analysis Agent (LangChain + Gemini)")
st.write("Enter organization information to generate a SWOT analysis:")

# API Key Input
api_key = st.text_input("Enter your Google API Key:", type="password") #type="password" hides the input

if not api_key:
    st.warning("Please enter your Google API Key.")
    st.stop()

# Initialize LangChain with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.7)

# Define the prompt template for SWOT analysis
prompt_template_swot = """
You are a management consultant and you have to analyze the following information and provide a detailed SWOT analysis of a company given by the
client.
Identify the Strengths, Weaknesses, Opportunities, and Threats.
Also give me the visual representation in cubicle form for the SWOT.

{context}

Format the response EXACTLY as follows, using clear headings and bullet points:

**Strengths:**
- [Strength 1]
- [Strength 2]
...

**Weaknesses:**
- [Weakness 1]
- [Weakness 2]
...

**Opportunities:**
- [Opportunity 1]
- [Opportunity 2]
...

**Threats:**
- [Threat 1]
- [Threat 2]
...
"""

prompt_swot = PromptTemplate(input_variables=["context"], template=prompt_template_swot)
llm_chain_swot = LLMChain(prompt=prompt_swot, llm=llm)

# Token encoder
encoder = tiktoken.get_encoding("cl100k_base")

# Function to generate SWOT analysis
def generate_swot(txt):
    response = llm_chain_swot.run(txt)
    return response

text_input = st.text_area("Enter organization information:")

if st.button("Generate SWOT Analysis"):
    with st.spinner('Generating SWOT analysis...'):
        swot_result = generate_swot(text_input)

        st.subheader("SWOT Analysis:")
        st.write(swot_result)

        # Calculate and display token counts
        query_tokens = len(encoder.encode(text_input))
        response_tokens = len(encoder.encode(swot_result))

        st.session_state.query_tokens += query_tokens
        st.session_state.response_tokens += response_tokens
        st.session_state.tokens_consumed += (query_tokens + response_tokens)

        st.sidebar.write(f"Total Tokens Consumed: {st.session_state.tokens_consumed}")
        st.sidebar.write(f"Query Tokens: {st.session_state.query_tokens}")
        st.sidebar.write(f"Response Tokens: {st.session_state.response_tokens}")

        print("Tokens consumed in this transaction...")
        print("Query token = ", query_tokens)
        print("Response tokens = ", response_tokens)
        st.session_state.tokens_consumed = 0
        st.session_state.query_tokens = 0
        st.session_state.response_tokens = 0
