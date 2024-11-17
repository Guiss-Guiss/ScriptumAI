import streamlit as st
from chroma_db_component import ChromaDBComponent
import logging
from language_utils import get_translation
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

def init_session_state():
    if 'delete_confirmation' not in st.session_state:
        st.session_state.delete_confirmation = False
    if 'file_to_delete' not in st.session_state:
        st.session_state.file_to_delete = None

def display_delete_confirmation(document_manager) -> None:
    confirm = st.warning(
        f"⚠️ {get_translation('confirm_delete_message')} {st.session_state.file_to_delete}?"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(get_translation('yes_delete'), key="confirm_yes"):
            try:
                document_manager.delete_file(st.session_state.file_to_delete)
                st.success(get_translation("file_deleted").format(
                    file_name=st.session_state.file_to_delete
                ))
                st.session_state.delete_confirmation = False
                st.session_state.file_to_delete = None
                st.rerun()
            except Exception as e:
                st.error(get_translation("delete_failed").format(
                    file_name=st.session_state.file_to_delete, 
                    error=str(e)
                ))
    
    with col2:
        if st.button(get_translation("cancel"), key="confirm_no"):
            st.session_state.delete_confirmation = False
            st.session_state.file_to_delete = None
            st.rerun()

class DocumentManager:
    def __init__(self, chroma_db: ChromaDBComponent):
        self.chroma_db = chroma_db

    def list_files(self) -> List[str]:
        try:
            documents = self.chroma_db.list_all_documents()
            file_set = {metadata['source'] for metadata in documents['metadatas']}
            return sorted(list(file_set))
        except Exception as e:
            logger.error(get_translation("error_listing_files").format(error=str(e)), exc_info=True)
            raise

    def delete_file(self, file_name: str) -> None:
        try:
            documents = self.chroma_db.list_all_documents()
            doc_ids_to_delete = [
                doc_id for doc_id, metadata in zip(documents['ids'], documents['metadatas']) 
                if metadata['source'] == file_name
            ]
            
            if doc_ids_to_delete:
                self.chroma_db.collection.delete(ids=doc_ids_to_delete)
                logger.info(get_translation("file_deleted_success").format(file_name=file_name))

                file_path = os.path.join("uploads", file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Physical file deleted: {file_path}")

                base_name = os.path.splitext(file_name)[0]
                processed_extensions = ['.txt', '_processed.txt']
                for ext in processed_extensions:
                    processed_path = os.path.join("uploads", f"{base_name}{ext}")
                    if os.path.exists(processed_path):
                        os.remove(processed_path)
                        logger.info(f"Processed file deleted: {processed_path}")

        except Exception as e:
            error_msg = str(e)
            logger.error(get_translation("error_deleting_file").format(
                file_name=file_name, 
                error=error_msg
            ), exc_info=True)
            raise RuntimeError(f"Failed to delete {file_name}: {error_msg}")

    def get_file_info(self, file_name: str) -> Dict:
        try:
            documents = self.chroma_db.list_all_documents()
            file_docs = [ 
                (doc_id, metadata) 
                for doc_id, metadata in zip(documents['ids'], documents['metadatas'])
                if metadata['source'] == file_name
            ]
            return {
                'count': len(file_docs),
                'exists_in_chroma': len(file_docs) > 0,
                'exists_in_uploads': os.path.exists(os.path.join("uploads", file_name))
            }
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {'error': str(e)}

def render_document_management(document_manager: DocumentManager) -> None:
    st.header(get_translation("document_management"))
    init_session_state()

    try:
        files = document_manager.list_files()
        if files:
            st.write(get_translation("uploaded_files"))
            for file_name in files:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(file_name)
                with col2:
                    file_info = document_manager.get_file_info(file_name)
                    if 'error' not in file_info:
                        st.write(f"{get_translation('chunks')} {file_info['count']}")
                with col3:
                    if st.button(get_translation("delete"), key=f"delete_{file_name}"):
                        st.session_state.delete_confirmation = True
                        st.session_state.file_to_delete = file_name
                        st.rerun()
                st.markdown("---")

            if st.session_state.delete_confirmation:
                display_delete_confirmation(document_manager)
        else:
            st.info(get_translation("no_files_found"))

    except Exception as e:
        st.error(get_translation("failed_to_list_files").format(error=str(e)))
        logger.error(get_translation("error_in_document_management").format(
            error=str(e)
        ), exc_info=True)

    st.markdown(f"""
    *{get_translation("note")}* {get_translation("warning_message")}
    """)