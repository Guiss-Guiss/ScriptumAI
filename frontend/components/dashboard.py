import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from typing import Dict, Any

def fetch_stats(api_url: str) -> Dict[str, Any]:
    """Fetch statistics from the API."""
    response = requests.get(f"{api_url}/stats")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching stats: {response.json().get('error', 'Unknown error')}")
        return {}

def render_dashboard(api_url: str):
    st.header("RAG System Dashboard")

    stats = fetch_stats(api_url)

    if not stats:
        st.warning("Unable to fetch system statistics. Please check the API connection.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", stats.get("total_documents", "N/A"))
    with col2:
        st.metric("Embedding Model", stats.get("embedding_model", "N/A"))
    with col3:
        st.metric("LLM Model", stats.get("llm_model", "N/A"))

    st.subheader("Supported File Types")
    file_types = stats.get("supported_file_types", [])
    st.write(", ".join(file_types) if file_types else "No file types available")

    doc_types = {
        "PDF": 50,
        "DOCX": 30,
        "TXT": 20,
    }
    df = pd.DataFrame(list(doc_types.items()), columns=["Type", "Count"])
    fig = px.pie(df, values="Count", names="Type", title="Document Types")
    st.plotly_chart(fig)


    st.subheader("Recent Activity")
    activity_data = [
        {"timestamp": "2024-10-10 14:30:00", "action": "Document Ingested", "details": "report.pdf"},
        {"timestamp": "2024-10-10 14:25:30", "action": "Query Processed", "details": "What is RAG?"},
        {"timestamp": "2024-10-10 14:20:00", "action": "Semantic Search", "details": "machine learning"},
    ]
    st.table(pd.DataFrame(activity_data))

    # System Health (dummy data for demonstration)
    st.subheader("System Health")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Usage", "45%")
    with col2:
        st.metric("Memory Usage", "60%")

    # Add a refresh button
    if st.button("Refresh Dashboard"):
        st.experimental_rerun()

