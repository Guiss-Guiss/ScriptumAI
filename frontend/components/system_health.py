import streamlit as st
import requests
import logging
from frontend.config import API_BASE_URL
from frontend.translations import get_text

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def fetch_health_status():
    try:
        logger.info(f"Attempting to fetch health status from {API_BASE_URL}/api/health")
        response = requests.get(f"{API_BASE_URL}/api/health")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response content: {response.text}")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch health status: {str(e)}", exc_info=True)
        st.error(f"Failed to fetch health status: {str(e)}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

def render_system_health(lang: str):
    logger.info(f"Rendering system health component with language: {lang}")
    st.subheader(get_text("system_health", lang))

    health_status = fetch_health_status()

    col1, col2 = st.columns(2)

    with col1:
        status_text = get_text(health_status['status'].lower(), lang)
        st.metric(get_text("overall_status", lang), status_text)

    with col2:
        db_status_text = get_text(health_status['database'].lower(), lang)
        st.metric(get_text("database_connectivity", lang), db_status_text)

    if health_status['status'] == 'unhealthy':
        st.error(get_text("error", lang).format(health_status.get('error', 'Unknown error')))
    else:
        st.success(get_text("all_systems_operational", lang))

    if st.button(get_text("refresh_health_status", lang)):
        st.rerun()
