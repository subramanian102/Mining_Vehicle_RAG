import streamlit as st
import os
from rag_engine import get_rag_chain

# 1. Page Configuration (Letting Streamlit handle native responsive layout padding)
st.set_page_config(page_title="FleetFix Workspace", page_icon="🚛", layout="wide")

# 2. Premium Professional UI Styling
st.markdown("""
    <style>
        /* Modern Minimalist Main App Background */
        .stApp {
            background-color: #FAFAFA;
        }
        
        /* Left Navigation Menu Styling */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #EAEAEA !important;
            padding-top: 20px;
        }
        
        /* Clean Custom Main Header Styling */
        .header-container {
            display: flex;
            align-items: center;
            padding-bottom: 15px;
            margin-bottom: 25px;
            border-bottom: 1px solid #EAEAEA;
        }
        .header-title {
            font-size: 24px;
            font-weight: 700;
            color: #1A1A1A;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
        }
        .header-badge {
            background-color: #EBF5FF;
            color: #0066CC;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 12px;
            margin-left: 15px;
        }

        /* Diagnostic Step List Cards - Premium Industrial Accent */
        .diagnostic-step {
            background-color: #FFFFFF;
            border-left: 4px solid #D45B00; /* Burnt Amber Accent Indicator */
            padding: 16px;
            margin: 12px 0;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            font-family: -apple-system, sans-serif;
            color: #2D2D2D;
            line-height: 1.6;
        }
        
        /* Expander Material Custom Borders */
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            border: 1px solid #EAEAEA !important;
            border-radius: 4px !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration (Clean & Compact)
with st.sidebar:
    st.markdown("### 💬 Operational Logs")
    sessions = ["Active Diagnostic Hub", "Engine Maintenance Overhaul", "Hydraulics Inspection Portfolio"]
    st.radio("Current Workspace:", sessions, index=0, label_visibility="collapsed")
    
    st.divider()
    st.markdown("### ⚙️ Engine Router")
    model_choice = st.selectbox("LLM Core Core:", ["gemini", "llama"], label_visibility="collapsed")
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Active Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. Clean Main Screen Header (Replaces the big orange bar)
st.markdown(f"""
    <div class="header-container">
        <div class="header-title">🚛 FleetFix Pro &nbsp;|&nbsp; <span style='font-weight:400; color:#666;'>Heavy Machinery Maintenance Copilot</span></div>
        <div class="header-badge">{model_choice.upper()} READY</div>
    </div>
""", unsafe_allow_html=True)

# Initialize Conversation History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Core Operational Logic Execution
if not os.path.exists("chroma_db"):
    st.error("⚠️ The vector database directory (`chroma_db`) was not detected. Please run ingestion setup first.")
else:
    @st.cache_resource(show_spinner="Connecting to Knowledge Base...")
    def load_pipeline(model_type):
        return get_rag_chain(persist_dir="chroma_db", model_type=model_type)
        
    try:
        rag_chain = load_pipeline(model_choice)
        
        # Render historical messages natively inside the main container layout grid
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                if msg.get("chunks"):
                    with st.expander("📑 View Extracted Manual Reference Context"):
                        for doc in msg["chunks"]:
                            st.markdown(f"**Source Manual Section:** `{doc['source']}`")
                            st.info(doc['text'])

        # 6. Bottom Pinned Chat Input Area (Completely responsive and standard-aligned)
        if user_query := st.chat_input("Enter breakdown symptoms or system fault codes..."):
            
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query, "chunks": []})
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing maintenance documentation chunks..."):
                    try:
                        from langchain_chroma import Chroma
                        from langchain_community.embeddings import HuggingFaceEmbeddings
                        
                        # Query vector DB to isolate specific documentation matches
                        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
                        db = Chroma(persist_directory="chroma_db", embedding_function=emb)
                        matched_docs = db.similarity_search(user_query, k=3)
                        
                        current_query_chunks = [
                            {"source": doc.metadata.get('source', 'Service_Manual_Part_A.pdf'), "text": doc.page_content}
                            for doc in matched_docs
                        ]
                        
                        # Generate RAG response
                        ai_response = rag_chain.invoke(user_query)
                        
                        # Convert steps into clean, crisp card containers
                        structured_output = ""
                        lines = ai_response.split('\n')
                        for line in lines:
                            cleaned_line = line.strip()
                            if cleaned_line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '- ', '* ')):
                                structured_output += f'<div class="diagnostic-step"><b>{cleaned_line}</b></div>'
                            elif cleaned_line:
                                structured_output += f'<p>{cleaned_line}</p>'
                        
                        st.markdown(structured_output, unsafe_allow_html=True)
                        
                        # Clean inline window reference documentation lookup
                        with st.expander("📑 View Extracted Manual Reference Context"):
                            for doc in current_query_chunks:
                                st.markdown(f"**Source Manual Section:** `{doc['source']}`")
                                st.info(doc['text'])
                        
                        # Save transaction frame to state history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": structured_output,
                            "chunks": current_query_chunks
                        })
                        
                    except Exception as err:
                        st.error(f"Runtime Diagnostics Generation Blocked: {err}")
                        
    except Exception as init_err:
        st.error(f"System Framework Core Connection Interrupted: {init_err}")