import streamlit as st
from main import create_chain, create_vectorstore, get_memory, ask_question

st.set_page_config(page_title="Fitness AI Coach", page_icon="🏋️", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("💬 Chats")

# Initialize sidebar history
if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = []

# ---------- NEW CHAT ----------
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_history_ui = []
    st.session_state.messages = []
    st.session_state.memory = get_memory()

# ---------- CLEAR CHAT ----------
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat_history_ui = []
    st.session_state.messages = []
    st.session_state.memory = get_memory()

# ---------- SHOW CHAT HISTORY ----------
for i, chat in enumerate(st.session_state.chat_history_ui):
    st.sidebar.markdown(f"**Chat {i+1}:** {chat[:30]}...")

# ---------- MAIN UI ----------
st.title("🏋️ Fitness AI Coach")

# ---------- VECTOR STORE ----------
if "vectorstore" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.vectorstore = create_vectorstore()

# ---------- CHAIN ----------
if "chain" not in st.session_state:
    st.session_state.chain = create_chain(st.session_state.vectorstore)

# ---------- MEMORY ----------
if "memory" not in st.session_state:
    st.session_state.memory = get_memory()

# ---------- CHAT STORAGE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- USER INPUT ----------
user_input = st.chat_input("Ask your fitness question...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)

    # Get response
    response = ask_question(
        st.session_state.chain,
        st.session_state.memory,
        user_input
    )

    # Show assistant response
    st.chat_message("assistant").markdown(response)

    # Save chat messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Save to sidebar history
    st.session_state.chat_history_ui.append(user_input)