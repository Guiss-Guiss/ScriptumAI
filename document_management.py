import streamlit as st
from chroma_db_component import ChromaDBComponent
import logging
from collections import defaultdict
from language_utils import get_translation

logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self, chroma_db: ChromaDBComponent):
        self.chroma_db = chroma_db

    def list_files(self):
        try:
            documents = self.chroma_db.list_all_documents()
            file_set = set()
            for metadata in documents['metadatas']:
                file_set.add(metadata['source'])
            return list(file_set)
        except Exception as e:
            logger.error(get_translation("error_listing_files").format(error=str(e)), exc_info=True)
            raise

    def delete_file(self, file_name):
        try:
            documents = self.chroma_db.list_all_documents()
            doc_ids_to_delete = [doc_id for doc_id, metadata in zip(documents['ids'], documents['metadatas']) if metadata['source'] == file_name]
            self.chroma_db.collection.delete(ids=doc_ids_to_delete)
            logger.info(get_translation("file_deleted_success").format(file_name=file_name))
        except Exception as e:
            logger.error(get_translation("error_deleting_file").format(file_name=file_name, error=str(e)), exc_info=True)
            raise

def render_document_management(document_manager):
    st.header(get_translation("document_management"))

    # Initialize session state for confirmation dialog
    if 'delete_confirmation' not in st.session_state:
        st.session_state.delete_confirmation = False
    if 'file_to_delete' not in st.session_state:
        st.session_state.file_to_delete = None

    try:
        files = document_manager.list_files()
        if files:
            st.write(get_translation("uploaded_files"))
            for file_name in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(file_name)
                with col2:
                    # When delete button is clicked, store the file name and show confirmation
                    if st.button(get_translation("delete"), key=f"delete_{file_name}"):
                        st.session_state.delete_confirmation = True
                        st.session_state.file_to_delete = file_name
                        st.rerun()
                st.markdown("---")

            # Show confirmation dialog if delete was clicked
            if st.session_state.delete_confirmation:
                confirm = st.warning(
                    f"{get_translation('confirm_delete_message')} {st.session_state.file_to_delete}?",
                    icon="⚠️"
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(get_translation("yes_delete"), key="confirm_yes"):
                        try:
                            document_manager.delete_file(st.session_state.file_to_delete)
                            st.success(get_translation("file_deleted").format(
                                file_name=st.session_state.file_to_delete))
                            # Reset the state
                            st.session_state.delete_confirmation = False
                            st.session_state.file_to_delete = None
                            st.rerun()
                        except Exception as e:
                            st.error(get_translation("delete_failed").format(
                                file_name=st.session_state.file_to_delete, 
                                error=str(e)))
                with col2:
                    if st.button(get_translation("cancel"), key="confirm_no"):
                        st.session_state.delete_confirmation = False
                        st.session_state.file_to_delete = None
                        st.rerun()

        else:
            st.info(get_translation("no_files_found"))
    except Exception as e:
        st.error(get_translation("failed_to_list_files").format(error=str(e)))
        logger.error(get_translation("error_in_document_management").format(error=str(e)), exc_info=True)

    st.markdown(get_translation("delete_note"))