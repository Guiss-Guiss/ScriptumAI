import streamlit as st
import requests
from config import API_BASE_URL
from datetime import datetime
from frontend.translations import get_text

def fetch_logs():
    try:
        url = f"{API_BASE_URL}/api/logs"

        response = requests.get(url)

        
        if response.headers.get('Content-Type', '').startswith('application/json'):
            return response.json()
        else:
            st.error(f"Unexpected response type. Expected JSON, got {response.headers.get('Content-Type', 'Unknown')}")
            return []
    except requests.RequestException as e:
        st.error(f"Failed to fetch logs: {str(e)}")
        return []
    except ValueError as e:
        st.error(f"Failed to parse JSON response: {str(e)}")
        return []
    
def render_system_logs(lang: str):
    st.subheader(get_text("system_logs", lang))

    logs = fetch_logs()

    if not logs:
        st.warning(get_text("no_logs_available", lang))
        return

    log_levels = list(set(log['level'] for log in logs))
    selected_levels = st.multiselect(get_text("filter_by_log_level", lang), log_levels, default=log_levels)

    date_range = st.date_input(get_text("select_date_range", lang), [datetime.now().date(), datetime.now().date()])

    search_term = st.text_input(get_text("search_logs", lang), "")

    filtered_logs = [
        log for log in logs
        if log['level'] in selected_levels
        and date_range[0] <= datetime.fromisoformat(log['timestamp']).date() <= date_range[1]
        and (search_term.lower() in log['message'].lower() or search_term.lower() in log['level'].lower())
    ]

    for log in filtered_logs:
        with st.expander(f"{log['timestamp']} - {log['level']}"):
            st.write(log['message'])

    if st.button(get_text("export_logs", lang)):
        export_logs(filtered_logs, lang)

def export_logs(logs, lang: str):
    log_text = "\n".join([f"{log['timestamp']} | {log['level']} | {log['message']}" for log in logs])
    try:
        st.download_button(
            label=get_text("download_logs", lang),
            data=log_text,
            file_name="system_logs.txt",
            mime="text/plain"
        )
        st.success(get_text("logs_exported_success", lang))
    except Exception as e:
        st.error(get_text("failed_to_export_logs", lang).format(str(e)))

