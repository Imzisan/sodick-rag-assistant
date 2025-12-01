"""
Streamlit Chat Interface for Sodick RAG Assistant
With AI-Generated Follow-up Questions
Run with: streamlit run app.py
"""

import streamlit as st
from src.helper import chat, clear_history, get_history_count, generate_followup_questions

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Sodick RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main .block-container {
        padding-bottom: 5rem;
    }
    footer {visibility: hidden;}
    
    /* Style for dynamic question section */
    .dynamic-questions {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# INITIALIZE SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

if "example_questions" not in st.session_state:
    # Starting with some default questions
    st.session_state.example_questions = [
        "What products does Sodick offer?",
        "Tell me about EDM machines",
        "What is the dielectric flushing requirement?",
        "How do I contact Sodick support?",
        "What industries does Sodick serve?"
    ]

if "questions_are_dynamic" not in st.session_state:
    st.session_state.questions_are_dynamic = False

# SIDEBAR

with st.sidebar:
    st.markdown("## Sodick RAG Assistant")
    st.markdown("---")
    
    # App information
    st.markdown("### About")
    st.markdown("""
    AI-powered assistant for Sodick Co., Ltd.
    
    **Features:**
    - Product information
    - Technical support
    - Documentation search
    - Conversational AI
    - Smart follow-ups
    """)
    
    st.markdown("---")
    
    # Show current session stats
    st.markdown("### Session Info")
    history_count = get_history_count()
    st.metric("Messages in History", history_count)
    st.metric("Chat Messages", len(st.session_state.messages))
    
    # Button to clear everything
    if st.button("Clear Chat History", use_container_width=True):
        clear_history()
        st.session_state.messages = []
        # Go back to default questions
        st.session_state.example_questions = [
            "What products does Sodick offer?",
            "Tell me about EDM machines",
            "What is the dielectric flushing requirement?",
            "How do I contact Sodick support?",
            "What industries does Sodick serve?"
        ]
        st.session_state.questions_are_dynamic = False
        st.rerun()
    
    st.markdown("---")
    
    # Technical details
    st.markdown("### Settings")
    st.info("""
    **Index:** sodic-en  
    **Model:** BAAI/bge-small-en-v1.5  
    **LLM:** OpenAI  
    **Retrieval:** Top-5 similarity
    """)
    
    st.markdown("---")
    
    # Question suggestions section
    if st.session_state.questions_are_dynamic:
        st.markdown("### Suggested Follow-ups")
        st.caption("AI-generated based on your conversation")
    else:
        st.markdown("### Example Questions")
        st.caption("Get started with these common questions")
    
    for q in st.session_state.example_questions:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            # Handle the click
            with st.spinner("Thinking..."):
                try:
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": q})
                    
                    # Get response from the system
                    response = chat(q)
                    
                    # Add assistant response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Generate new follow-up questions
                    st.session_state.example_questions = generate_followup_questions(q, response)
                    st.session_state.questions_are_dynamic = True
                    
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.75rem; padding: 0.5rem;'>
        <strong>Powered by:</strong><br>
        LangChain + Pinecone + OpenAI<br><br>
        <strong>Built for:</strong><br>
        Sodick Co., Ltd.<br><br>
        <a href='https://www.sodick.co.jp/en/contact' target='_blank' style='color: #1f77b4;'>
        Contact Sodick
        </a>
    </div>
    """, unsafe_allow_html=True)

# MAIN CHAT INTERFACE

# Page header
st.markdown('<div class="main-header">Sodick RAG Assistant</div>', unsafe_allow_html=True)
st.markdown("Ask me anything about Sodick products, services, and technical support!")

# Let users know when we're showing smart suggestions
if st.session_state.questions_are_dynamic:
    st.info("Smart Mode Active: Sidebar shows AI-generated follow-up questions based on your conversation!")

st.markdown("---")

# Show all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input (always visible)
user_input = st.chat_input("Type your question here...", key="chat_input")

# Handle new user input
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show user message right away
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chat(user_input)
                st.markdown(response)
                
                # Save assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Generate follow-up questions in the background
                with st.spinner("Generating follow-up questions..."):
                    st.session_state.example_questions = generate_followup_questions(user_input, response)
                    st.session_state.questions_are_dynamic = True
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
