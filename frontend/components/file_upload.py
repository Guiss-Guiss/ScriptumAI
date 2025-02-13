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
    if 'ingestion_tasks' not in st.session_state:
        st.session_state.ingestion_tasks = {}

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
            st.session_state.ingestion_tasks[file.name] = task_id
            return check_ingestion_status(task_id, file.name, lang)
        else:
            error_message = response.json().get('error', 'Unknown error occurred')
            st.error(f"Failed to start ingestion for {file.name}: {error_message}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error ingesting {file.name}: {str(e)}", exc_info=True)
        st.error(f"Error ingesting {file.name}: {str(e)}")
        return False


def check_ingestion_status(task_id: str, file_name: str, lang: str) -> bool:
    max_retries = 10
    retry_count = 0

    with requests.Session() as session:
        while retry_count < max_retries:
            try:
                response = session.get(f"{API_BASE_URL}/api/ingestion_status/{task_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get('status', '')

                    if status == "Completed":
                        st.success(f"{get_text('ingestion_success', lang)}: {file_name}")
                        if file_name not in st.session_state.ingested_files:
                            st.session_state.ingested_files.append(file_name)
                            save_ingested_files(st.session_state.ingested_files)
                        return True
                    elif status.startswith("Failed"):
                        st.error(f"{get_text('ingestion_failed', lang)} {file_name}: {status}")
                        return False
                retry_count += 1
                time.sleep(2)
            except requests.RequestException as e:
                logger.error(f"Error checking status for {file_name}: {str(e)}", exc_info=True)
                retry_count += 1
                time.sleep(5)
        st.error(f"Ingestion for {file_name} timed out.")
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

    # st.subheader(get_text("or_ingest_directory", lang))
    # dir_path = st.text_input(get_text("enter_directory_path", lang))
    # if dir_path and st.button(get_text("ingest_directory", lang)):
    #     if not os.path.isdir(dir_path):
    #         st.error(get_text("invalid_directory", lang))
    #     else:
    #         process_directory(dir_path, supported_types, lang)

    st.subheader(get_text("supported_file_types", lang))
    st.write(", ".join(supported_types))

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
        queue_status = st.empty()
        current_file_status = st.empty()
        progress_bar = st.progress(0)

        # Traiter les fichiers séquentiellement
        for i, file in enumerate(files):
            try:
                queue_status.info(f"File {i+1}/{len(files)}: {file.name}")
                current_file_status.write(f"Processing: {file.name}")
                
                if ingest_file(file, lang):
                    successful_ingests += 1
                    current_file_status.success(f"Completed: {file.name}")
                else:
                    failed_ingests += 1
                    current_file_status.error(f"Failed: {file.name}")
                
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {str(e)}", exc_info=True)
                failed_ingests += 1
                current_file_status.error(f"Error with {file.name}: {str(e)}")
                
            finally:
                # Mettre à jour la barre de progression globale
                progress_bar.progress((i + 1) / len(files))

        # Résumé final
        if successful_ingests > 0:
            st.success(f"Successfully processed {successful_ingests} files")
        if failed_ingests > 0:
            st.error(f"Failed to process {failed_ingests} files")

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

