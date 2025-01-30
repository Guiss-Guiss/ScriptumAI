import streamlit as st
import json
import os
from file_uploader import file_uploader
from chroma_db_component import ChromaDBComponent
from embedding_component import EmbeddingComponent
from retrieval_component import RetrievalComponent
from model_selector import render_model_selector
from query_component import QueryComponent
from retrieval_system import RetrievalSystem
from rag_component import RAGComponent
from document_management import DocumentManager, render_document_management
from language_utils import get_translation, init_session_state, change_language, get_current_language
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Script started")

init_session_state()

def load_preferences():
    if os.path.exists('user_preferences.json'):
        with open('user_preferences.json', 'r') as f:
            prefs = json.load(f)
        change_language(prefs.get('language', 'en'))
        return prefs
    return {'n_results': 5, 'confidence_threshold': 0.7, 'language': 'en'}

def save_preferences(preferences):
    with open('user_preferences.json', 'w') as f:
        json.dump(preferences, f)
    change_language(preferences['language'])

async def main():
    current_language = get_current_language()
    try:
        logger.info("Entering main function")
        st.title(get_translation("title"))
        logger.info(get_translation("app_started"))

        preferences = load_preferences()

        try:
            chroma_db = ChromaDBComponent()
            embedding_component = EmbeddingComponent()
            retrieval_component = RetrievalComponent(chroma_db, embedding_component)
            query_component = QueryComponent()
            retrieval_system = RetrievalSystem(retrieval_component, query_component)
            rag_component = RAGComponent()
            document_manager = DocumentManager(chroma_db)
        except Exception as e:
            st.error(get_translation("component_init_error").format(error=str(e)))
            logger.error(f"Component initialization error: {str(e)}", exc_info=True)
            return

        st.sidebar.title(get_translation("navigation"))
        page = st.sidebar.radio(get_translation("go_to"), [
            get_translation("upload"),
            get_translation("search"),
            get_translation("manage_documents"),
            get_translation("settings")
        ])

        if page == get_translation("upload"):
            st.header(get_translation("document_upload"))
            uploaded_files = st.file_uploader(get_translation("choose_files"), accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'html', 'md'])

            if uploaded_files:
                try:
                    logger.info(get_translation("files_uploaded").format(files=", ".join([file.name for file in uploaded_files])))
                    st.info(get_translation("processing_files"))
                    results = await file_uploader(uploaded_files)
                    for result in results:
                        st.write(result)
                    st.success(get_translation("files_processed"))
                    logger.info(get_translation("file_processing_completed"))
                except Exception as e:
                    st.error(get_translation("file_processing_error").format(error=str(e)))
                    logger.error(f"File processing error: {str(e)}", exc_info=True)
            else:
                logger.info(get_translation("no_files_uploaded"))

        elif page == get_translation("search"):
            st.header(get_translation("search_qa"))
            
            # Add model selector - now synchronous
            selected_model = render_model_selector()
            
            # Initialize RAG component with selected model
            if 'rag_component' not in st.session_state or st.session_state.current_model != selected_model:
                st.session_state.rag_component = RAGComponent(model_name=selected_model)
                st.session_state.current_model = selected_model
            
            query = st.text_input(get_translation("enter_query"))
            
            n_results = st.slider(get_translation("num_results"), 1, 1000, preferences['n_results'])
            confidence_threshold = st.slider(get_translation("confidence_threshold"), 0.0, 1.0, preferences['confidence_threshold'])

            search_button = st.button(get_translation("search_answer"))

            if search_button and query:
                try:
                    with st.spinner(get_translation("searching")):
                        relevant_chunks = await retrieval_system.fetch_relevant_chunks(query, n_results)
                        
                        filtered_chunks = [chunk for chunk in relevant_chunks if isinstance(chunk, dict) and chunk.get('similarity_score', 0) >= confidence_threshold]
                        
                        if filtered_chunks:
                            answer = await st.session_state.rag_component.generate_answer(query, filtered_chunks)
                            
                            st.subheader(get_translation("generated_answer"))
                            st.write(answer)
                            
                            st.subheader(get_translation("relevant_chunks"))
                            for i, chunk in enumerate(filtered_chunks, 1):
                                with st.expander(f"{get_translation('chunk')} {i}"):
                                    st.write(f"{get_translation('chunk_content')} {chunk['content'][:200]}...")
                                    st.write(f"{get_translation('chunk_similarity')} {chunk['similarity_score']:.2f}")
                                    st.write(f"{get_translation('chunk_metadata')} {chunk['metadata']}")
                        else:
                            st.warning(get_translation("no_relevant_docs"))
                except Exception as e:
                    st.error(get_translation("query_processing_error").format(error=str(e)))
                    logger.error(f"Query processing error: {str(e)}", exc_info=True)
        
        elif page == get_translation("manage_documents"):
            render_document_management(document_manager)

        elif page == get_translation("settings"):
            st.header(get_translation("user_preferences"))
            
            language_names = {
                "en": "English",
                "fr": "Français",
                "es": "Español"
            }
            
            current_language = get_current_language()
            
            selected_language = st.selectbox(
                get_translation("language_selection"),
                options=list(language_names.keys()),
                format_func=lambda x: language_names[x],
                index=list(language_names.keys()).index(current_language)
            )
            
            new_n_results = st.slider(get_translation("default_number_of_results"), 1, 100, preferences['n_results'])
            new_confidence_threshold = st.slider(get_translation("default_confidence_threshold"), 0.0, 1.0, preferences['confidence_threshold'])
            
            if st.button(get_translation("save_preferences")):
                new_preferences = {
                    'n_results': new_n_results, 
                    'confidence_threshold': new_confidence_threshold,
                    'language': selected_language
                }
                save_preferences(new_preferences)
                st.success(get_translation("preferences_saved"))
                
                st.info(get_translation("refresh_to_apply"))                
    except Exception as e:
        st.error(get_translation("unexpected_error").format(error=str(e)))
        logger.error(f"Unexpected error in main function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        logger.info(get_translation("starting_app"))
        asyncio.run(main())
        logger.info(get_translation("app_completed"))
    except Exception as e:
        st.error(get_translation("app_runtime_error", error=str(e)))
        logger.error(f"Application runtime error: {str(e)}", exc_info=True)
