#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Aditya Gupta, FT251006, Section-1

#Objective: Using Streamlit and LangChain design an AI Agent that takes as input information about an 

# organization such as NovaEdge (below) and emits a detailed SWOT analysis. The Analysis should 

# include a visual representation of the key points of the SWOT analysis. 

# Import necessary libraries
import os
import streamlit as st #import streamlit
from langchain_google_genai import ChatGoogleGenerativeAI #import gemini library
from langchain.prompts import PromptTemplate #import langchain
from langchain.chains import LLMChain #import langchain
import tiktoken  # to count the tokens
#from dotenv import load_dotenv

#load_dotenv()

# Fetch API key from environment variable
#GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set. Please set it in your environment variables.")  # Display error message if API key is not found
    st.stop()  # Stop execution if API key is not found

# Initialize token counters in Streamlit session state
if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0  # Initialize total tokens consumed
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0  # Initialize query tokens
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0  # Initialize response tokens

# Initialize LangChain with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.7)  # Initialize the language model

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

prompt_swot = PromptTemplate(input_variables=["context"], template=prompt_template_swot)  # Create a prompt template
llm_chain_swot = LLMChain(prompt=prompt_swot, llm=llm)  # Create an LLM chain for SWOT analysis

# Token encoder
encoder = tiktoken.get_encoding("cl100k_base")  # Get the token encoder

# Function to generate SWOT analysis
def generate_swot(txt):
    response = llm_chain_swot.run(txt)  # Run the LLM chain to generate SWOT analysis
    return response  # Return the SWOT analysis

# Streamlit UI Setup
st.set_page_config(page_title="SWOT Analysis Agent (LangChain + Gemini)")  # Set the page title
st.title("SWOT Analysis Agent (LangChain + Gemini)")  # Set the page title
st.write("Enter organization information to generate a SWOT analysis:")  # Display instructions

text_input = st.text_area("Enter organization information:")  # Create a text area for user input

if st.button("Generate SWOT Analysis"):  # Create a button to trigger SWOT analysis
    with st.spinner('Generating SWOT analysis...'):  # Display a spinner while generating SWOT analysis
        swot_result = generate_swot(text_input)  # Generate SWOT analysis

        st.subheader("SWOT Analysis:")  # Display the SWOT analysis heading
        st.write(swot_result)  # Display the SWOT analysis

        # Calculate and display token counts
        query_tokens = len(encoder.encode(text_input))  # Calculate query tokens
        response_tokens = len(encoder.encode(swot_result))  # Calculate response tokens

        st.session_state.query_tokens += query_tokens  # Update query tokens in session state
        st.session_state.response_tokens += response_tokens  # Update response tokens in session state
        st.session_state.tokens_consumed += (query_tokens + response_tokens)  # Update total tokens consumed in session state

        st.sidebar.write(f"Total Tokens Consumed: {st.session_state.tokens_consumed}")  # Display total tokens consumed
        st.sidebar.write(f"Query Tokens: {st.session_state.query_tokens}")  # Display query tokens
        st.sidebar.write(f"Response Tokens: {st.session_state.response_tokens}")  # Display response tokens

        print("Tokens consumed in this transaction...")  # Print tokens consumed in this transaction
        print("Query token = ", query_tokens)  # Print query tokens
        print("Response tokens = ", response_tokens)  # Print response tokens
        st.session_state.tokens_consumed = 0  # Reset total tokens consumed
        st.session_state.query_tokens = 0  # Reset query tokens
        st.session_state.response_tokens = 0  # Reset response tokens


# In[ ]:




