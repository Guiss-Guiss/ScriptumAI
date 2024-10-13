import streamlit as st
import requests
from typing import Callable, Dict, Any

def render_search_bar(api_url: str, on_result: Callable[[str, Dict[str, Any]], None]):
    st.header("Search and Query")

    query = st.text_input("Enter your search query or question", key="search_query")


    search_type = st.radio("Select search type", ["RAG Query", "Semantic Search"])

    with st.expander("Advanced Options"):
        if search_type == "Semantic Search":
            k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                help="Controls randomness in response generation. Higher values make output more random.")
        
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=150, step=50,
                               help="Maximum number of tokens in the response.")

    if st.button("Search / Query"):
        if not query:
            st.warning("Please enter a search query or question.")
            return

        with st.spinner("Processing..."):
            try:
                if search_type == "RAG Query":
                    response = requests.post(f"{api_url}/query", json={
                        "query": query,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    })
                else: 
                    response = requests.post(f"{api_url}/search", json={
                        "query": query,
                        "k": k
                    })

                if response.status_code == 200:
                    result = response.json()
                    on_result(search_type.lower().replace(" ", "_"), result)
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")
            except requests.RequestException as e:
                st.error(f"Network error: {str(e)}")

    st.subheader("Recent Searches/Queries")
    recent_searches = st.session_state.get("recent_searches", [])
    for recent_query in recent_searches:
        if st.button(recent_query, key=f"recent_{recent_query}"):
            st.session_state.search_query = recent_query

def update_recent_searches(query: str):
    recent_searches = st.session_state.get("recent_searches", [])
    if query not in recent_searches:
        recent_searches.insert(0, query)
        recent_searches = recent_searches[:5]
        st.session_state.recent_searches = recent_searches


