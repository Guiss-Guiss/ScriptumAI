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
    st.header(get_text("system_logs", lang))

    # Fetch logs
    logs = fetch_logs()

    if not logs:
        st.warning(get_text("no_logs_available", lang))
        return

    # Log level filter
    log_levels = ["Choose an option", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    selected_level = st.selectbox(
        get_text("filter_by_log_level", lang),
        options=log_levels,
        index=0  # This sets "Choose an option" as the default
    )

    # Date range filter
    start_date = st.date_input(f"{get_text('select_date_range', lang)} - Start")
    end_date = st.date_input(f"{get_text('select_date_range', lang)} - End")

    # Search filter
    search_term = st.text_input(get_text("search_logs", lang))

    # Apply filters
    filtered_logs = logs
    if selected_level != "Choose an option":
        filtered_logs = [log for log in filtered_logs if log['level'] == selected_level]
    filtered_logs = [log for log in filtered_logs if start_date <= datetime.fromisoformat(log['timestamp']).date() <= end_date]
    if search_term:
        filtered_logs = [log for log in filtered_logs if search_term.lower() in log['message'].lower()]

    # Display filtered logs
    for log in filtered_logs:
        st.text(f"{log['timestamp']} - {log['level']}: {log['message']}")

    # Export logs button
    if st.button(get_text("export_logs", lang)):
        csv = logs_to_csv(filtered_logs)
        st.download_button(
            label=get_text("download_logs", lang),
            data=csv,
            file_name="system_logs.csv",
            mime="text/csv"
        )

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

