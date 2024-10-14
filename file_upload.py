import streamlit as st
import requests
import time
from typing import List
import os
import json
from frontend.config import API_BASE_URL
import logging
from frontend.translations import get_text

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

INGESTED_FILES_PATH = "frontend/ingested_files.json"

def load_ingested_files():
    if os.path.exists(INGESTED_FILES_PATH):
        with open(INGESTED_FILES_PATH, "r") as f:
            return json.load(f)
    return []

def save_ingested_files(files):
    with open(INGESTED_FILES_PATH, "w") as f:
        json.dump(files, f)

def init_session_state():
    if 'ingested_files' not in st.session_state:
        st.session_state.ingested_files = load_ingested_files()

def ingest_file(file, lang: str) -> bool:
    try:
        logger.debug(f"Preparing to send file: {file.name}, type: {file.type}")
        files = {"file": (file.name, file.getvalue(), file.type)}
        logger.debug(f"Sending POST request to {API_BASE_URL}/api/ingest")
        response = requests.post(f"{API_BASE_URL}/api/ingest", files=files)
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response content: {response.text}")

        if response.status_code == 202:
            task_id = response.json().get('task_id')
            return check_ingestion_status(task_id, file.name, lang)
        else:
            error_message = response.json().get('error', 'Unknown error occurred')
            st.error(f"Failed to start ingestion for {file.name}: {error_message}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error ingesting {file.name}: {str(e)}", exc_info=True)
        st.error(f"Error ingesting {file.name}: {str(e)}")
        return False

def check_ingestion_status(task_id: str, file_name: str, lang: str, max_retries: int = 999999) -> bool:
    status_placeholder = st.empty()
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/api/ingestion_status/{task_id}")
            if response.status_code == 200:
                status = response.json().get('status')
                logger.debug(f"Status for {file_name}: {status}")

                if status == "Completed":
                    status_placeholder.success(f"{get_text('ingestion_success', lang)}: {file_name}")
                    if file_name not in st.session_state.ingested_files:
                        st.session_state.ingested_files.append(file_name)
                        save_ingested_files(st.session_state.ingested_files)
                    return True
                elif status.startswith("Failed"):
                    status_placeholder.error(f"{get_text('ingestion_failed', lang)} {file_name}: {status}")
                    return False
                else:
                    status_placeholder.info(f"{get_text('ingestion_in_progress', lang)}: {file_name} ({i+1} sec.)")
            else:
                st.error(f"Error: Received status code {response.status_code} from the ingestion API for {file_name}.")
                return False

            time.sleep(1)

        except requests.RequestException as e:
            logger.error(f"Error checking ingestion status for {file_name}: {str(e)}", exc_info=True)
            status_placeholder.error(f"Error checking status for {file_name}: {str(e)}")
            return False

    status_placeholder.error(f"{get_text('ingestion_timeout', lang)}: {file_name}")
    return False

def render_file_upload(supported_types: List[str], lang: str):
    init_session_state()
    st.header(get_text("ingest_documents", lang))


    uploaded_files = st.file_uploader(get_text("choose_files", lang),
                                      type=supported_types,
                                      accept_multiple_files=True)

    if uploaded_files:
        st.write(get_text("selected_files", lang).format(len(uploaded_files)))
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.type})")
            logger.debug(f"File selected for upload: {file.name}, type: {file.type}")

        if st.button(get_text("ingest_selected", lang)):
            process_uploads(uploaded_files, lang)

    # Directory uploader
    st.subheader(get_text("or_ingest_directory", lang))
    dir_path = st.text_input(get_text("enter_directory_path", lang))
    if dir_path and st.button(get_text("ingest_directory", lang)):
        if not os.path.isdir(dir_path):
            st.error(get_text("invalid_directory", lang))
        else:
            process_directory(dir_path, supported_types, lang)

    # Display supported file types
    st.subheader(get_text("supported_file_types", lang))
    st.write(", ".join(supported_types))

    # Display list of ingested files
    st.subheader(get_text("ingested_files", lang))
    if st.session_state.ingested_files:
        for file in st.session_state.ingested_files:
            st.write(f"- {file}")
    else:
        st.write(get_text("no_ingested_files", lang))

    if st.button(get_text("refresh_ingested_files", lang)):
        st.session_state.ingested_files = load_ingested_files()
        st.rerun()

def process_uploads(files, lang: str):
    with st.spinner(get_text("ingesting_documents", lang)):
        successful_ingests = 0
        failed_ingests = 0

        progress_bar = st.progress(0)
        for i, file in enumerate(files):
            if ingest_file(file, lang):
                successful_ingests += 1
            else:
                failed_ingests += 1
            progress_bar.progress((i + 1) / len(files))

        st.write(get_text("ingestion_complete", lang).format(successful_ingests, failed_ingests))

def process_directory(dir_path: str, supported_types: List[str], lang: str):
    with st.spinner(get_text("ingesting_directory", lang)):
        successful_ingests = 0
        failed_ingests = 0
        failed_files = []

        files_to_process = [os.path.join(root, file) for root, _, files in os.walk(dir_path) for file in files]
        total_files = len(files_to_process)

        progress_bar = st.progress(0)

        for i, file_path in enumerate(files_to_process):
            file_type = os.path.splitext(file_path)[1][1:].lower()

            if file_type in supported_types:
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()

                    class SimpleFile:
                        def __init__(self, name, content, content_type):
                            self.name = name
                            self.content = content
                            self.type = content_type

                        def getvalue(self):
                            return self.content

                    simple_file = SimpleFile(os.path.basename(file_path), file_content, f"application/{file_type}")

                    if ingest_file(simple_file, lang):
                        successful_ingests += 1
                    else:
                        failed_ingests += 1
                        failed_files.append(file_path)

                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}", exc_info=True)
                    failed_ingests += 1
                    failed_files.append(file_path)
            else:
                logger.warning(f"Skipping unsupported file type: {file_path}")

            progress_bar.progress((i + 1) / total_files)

        st.success(get_text("successful_ingestion", lang).format(successful_ingests))
        if failed_ingests > 0:
            st.warning(get_text("failed_ingestion", lang).format(failed_ingests))
            for file in failed_files:
                st.write(f"- {file}")

