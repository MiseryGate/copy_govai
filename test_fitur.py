import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from getpass import getpass
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent

#Read Data
data = './data_final_ekonomi.csv'

groq_api = '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'
llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)

# openai = OpenAI(temperature=0.7, openai_api_key=openai_api_key, model="gpt-3.5-turbo")
# agent = create_pandas_dataframe_agent(openai, data, verbose=True, allow_dangerous_code=True)
agent = create_csv_agent(llm, data, verbose=True, allow_dangerous_code=True, max_execution_time=100000000)

st.title("Test Fitur")

# Initialize the session state messages if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Get user input
prompt = st.chat_input("Say something")
if prompt:
    # Display the user's message
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    response = agent.invoke(prompt)
    # Display the assistant's response
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
