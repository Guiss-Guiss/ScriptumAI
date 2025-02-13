"""
RAG Application ScriptumAI
Copyright (C) 2024 Guillaume Ste-Marie

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import streamlit as st
import requests
import logging
import sys
import os

from frontend.components.system_health import render_system_health
from frontend.components.system_logs import render_system_logs
from frontend.components.file_upload import render_file_upload
from frontend.components.results_display import render_results
from frontend.config import API_BASE_URL
from frontend.translations import get_text
from frontend.language_utils import render_language_selector, get_user_language


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Current Python path in app.py: {sys.path}")
logger.debug(f"Current working directory: {os.getcwd()}")

st.set_page_config(page_title="ScriptumAI", page_icon="📚", layout="wide")

current_lang = get_user_language()

menu_items = [
    "home",
    "ingest_documents",
    "query",
    "semantic_search",
    "system_statistics",
    "system_health",
    "system_logs"
]
translated_menu = [get_text(item, current_lang) for item in menu_items]

choice = st.sidebar.selectbox(get_text("menu", current_lang), translated_menu)

st.title(get_text("app_title", current_lang))

choice_key = menu_items[translated_menu.index(choice)]

if choice_key == "home":
    render_language_selector(get_text)
    st.subheader(get_text("welcome_message", current_lang))
    st.write(get_text("navigation_instruction", current_lang))

elif choice_key == "ingest_documents":
    render_file_upload(["txt", "pdf", "docx", "md"], current_lang)

elif choice_key == "query":
    st.header(get_text("process_query", current_lang))
    query = st.text_input(get_text("enter_query", current_lang))
    if st.button(get_text("process_query", current_lang)):
        with st.spinner(get_text("processing_query", current_lang)):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/query",
                    json={"query": query},
                    headers={"Content-Type": "application/json"}
                )
                logger.debug(f"API Response status: {response.status_code}")
                logger.debug(f"API Response content: {response.text[:200]}...")
                
                if response.status_code == 200:
                    result = response.json()
                    render_results("query", result, current_lang)
                else:
                    error_msg = response.json().get('error', 'Unknown error occurred') if response.content else 'Empty response'
                    st.error(get_text("error_processing_query", current_lang).format(error_msg))
            except requests.RequestException as e:
                logger.error(f"Network error while processing query: {e}", exc_info=True)
                st.error(get_text("network_error", current_lang).format(str(e)))

elif choice_key == "semantic_search":
    st.header(get_text("semantic_search", current_lang))
    query = st.text_input(get_text("enter_search_query", current_lang))
    k = st.slider(get_text("number_of_results", current_lang), min_value=1, max_value=20, value=5)
    if st.button(get_text("search", current_lang)):
        with st.spinner(get_text("searching", current_lang)):
            try:
                response = requests.post(f"{API_BASE_URL}/search", json={"query": query, "k": k})
                if response.status_code == 200:
                    results = response.json()
                    render_results("search", results, current_lang)
                else:
                    st.error(get_text("error_performing_search", current_lang).format(response.json().get('error', 'Unknown error occurred')))
            except requests.RequestException as e:
                logger.error(f"Network error while performing search: {e}", exc_info=True)
                st.error(get_text("network_error", current_lang).format(str(e)))

elif choice_key == "system_statistics":
    st.header(get_text("system_statistics", current_lang))
    if st.button(get_text("get_stats", current_lang)):
        with st.spinner(get_text("fetching_stats", current_lang)):
            try:
                response = requests.get(f"{API_BASE_URL}/stats")
                if response.status_code == 200:
                    stats = response.json()
                    for key, value in stats.items():
                        if isinstance(value, list):
                            st.subheader(key)
                            st.write(", ".join(map(str, value)))
                        else:
                            st.metric(label=key, value=value)
                else:
                    st.error(get_text("error_fetching_stats", current_lang).format(response.json().get('error', 'Unknown error occurred')))
            except requests.RequestException as e:
                logger.error(f"Network error while fetching stats: {e}", exc_info=True)
                st.error(get_text("network_error", current_lang).format(str(e)))

elif choice_key == "system_health":
    render_system_health(current_lang)

elif choice_key == "system_logs":
    render_system_logs(current_lang)