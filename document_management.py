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

    try:
        files = document_manager.list_files()
        if files:
            st.write(get_translation("uploaded_files"))
            for file_name in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(file_name)
                with col2:
                    if st.button(get_translation("delete"), key=f"delete_{file_name}"):
                        if st.button(get_translation("confirm_delete"), key=f"confirm_delete_{file_name}"):
                            try:
                                document_manager.delete_file(file_name)
                                st.success(get_translation("file_deleted").format(file_name=file_name))
                                st.rerun()
                            except Exception as e:
                                st.error(get_translation("delete_failed").format(file_name=file_name, error=str(e)))
                st.markdown("---")
        else:
            st.info(get_translation("no_files_found"))
    except Exception as e:
        st.error(get_translation("failed_to_list_files").format(error=str(e)))
        logger.error(get_translation("error_in_document_management").format(error=str(e)), exc_info=True)

    st.markdown(get_translation("delete_note"))