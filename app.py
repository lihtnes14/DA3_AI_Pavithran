import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model_name="gemma2-9b-it"
)

# Define State for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Node logic
def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Streamlit Page Configuration
st.set_page_config(page_title="MoneyMentor", layout="wide")

# 💅 Updated CSS Styling
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(to right, #e3f2fd, #fce4ec);
        color: #212121;
    }

    .chat-container {
        max-width: 850px;
        margin: 0 auto;
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        padding: 30px;
    }

    .stChatMessage {
        padding: 14px;
        border-radius: 12px;
        margin: 6px 0;
        font-size: 15px;
    }

    .stChatMessage.user {
        background-color: #e1f5fe;
        text-align: left;
    }

    .stChatMessage.assistant {
        background-color: #ede7f6;
        text-align: right;
    }

    .stTextInput > div > input {
        border-radius: 30px !important;
        padding: 14px 20px;
        font-size: 15px;
        border: 1px solid #bbb;
        background-color: #ffffff;
    }

    .stButton > button {
        border-radius: 30px;
        padding: 12px 30px;
        background-color: #1e88e5;
        color: white;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #1565c0;
    }

    h1 {
        font-size: 2.8rem;
        color: #1e88e5;
        text-align: center;
    }

    h2 {
        font-size: 1.4rem;
        color: #424242;
        text-align: center;
        margin-bottom: 20px;
    }

    .description {
        text-align: center;
        font-size: 1rem;
        color: #616161;
        margin-bottom: 35px;
    }
    </style>
""", unsafe_allow_html=True)

# 🚀 App Title and Description
st.title("MoneyMentor 💰")
st.markdown("<h2>Your Smart AI Companion for Financial Guidance</h2>", unsafe_allow_html=True)
st.markdown('<p class="description">Whether you need help understanding loans, budgeting tips, or financial planning, I’m here to assist you.</p>', unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
def display_chat_history():
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

display_chat_history()

# Input
user_input = st.text_input("Type your message here...", placeholder="Ask about credit scores, investments, or savings tips...", key="chat_input")

# Response Handling
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    full_response = ""
    with st.chat_message("assistant"):
        response_box = st.empty()
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                if "messages" in value:
                    msg = value["messages"]
                    full_response = msg.content
                    response_box.markdown(full_response)

    st.session_state.chat_history.append(("assistant", full_response))

# Clear button
if st.button("Clear Chat History", key="clear_button", use_container_width=True):
    st.session_state.chat_history.clear()
    st.experimental_rerun()
