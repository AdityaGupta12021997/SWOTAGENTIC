import os
import time
import streamlit as st
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from google.api_core import exceptions

# --- API Configuration ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

# --- Initialize Model ---
# Using 1.5-flash as it has the highest RPD (Requests Per Day) for the Free Tier
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=api_key, 
    temperature=0.7
)

# --- SWOT Template ---
prompt_template_swot = """
You are a management consultant. Analyze the following information and provide a detailed SWOT analysis.
Identify Strengths, Weaknesses, Opportunities, and Threats.

{context}

Format the response EXACTLY as follows:

**Strengths:**
- [Point]
...
**Weaknesses:**
- [Point]
...
**Opportunities:**
- [Point]
...
**Threats:**
- [Point]
...

Finally, provide a Markdown Table summarizing the SWOT.
"""

prompt_swot = PromptTemplate(input_variables=["context"], template=prompt_template_swot)
chain = prompt_swot | llm

# --- UI Setup ---
st.set_page_config(page_title="SWOT Agent (Quota Optimized)", layout="wide")
st.title("🚀 SWOT Analysis Agent")
st.info("Note: Using Gemini 1.5 Flash to minimize 'Quota Exceeded' errors.")

# Token encoder
encoder = tiktoken.get_encoding("cl100k_base")

text_input = st.text_area("Enter organization information:", height=250)

if st.button("Generate SWOT Analysis"):
    if not text_input:
        st.warning("Please enter some text first.")
    else:
        with st.spinner('Consulting Gemini (with retry logic)...'):
            # --- Robust Execution with Retries ---
            max_retries = 3
            retry_delay = 10 # Seconds to wait on 429 error
            
            for attempt in range(max_retries):
                try:
                    response = chain.invoke({"context": text_input})
                    swot_result = response.content
                    
                    # Display Results
                    st.success("Analysis Complete!")
                    st.markdown(swot_result)
                    
                    # Log tokens
                    q_tokens = len(encoder.encode(text_input))
                    r_tokens = len(encoder.encode(swot_result))
                    st.sidebar.metric("Tokens Used", q_tokens + r_tokens)
                    break # Success! Exit the retry loop
                
                except exceptions.ResourceExhausted:
                    if attempt < max_retries - 1:
                        st.warning(f"Quota hit. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2 # Exponential backoff
                    else:
                        st.error("Maximum retries reached. The Google Free Tier limit is strictly enforced right now. Please wait 1-2 minutes before trying again.")
                
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    break
