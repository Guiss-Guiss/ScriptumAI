import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from frontend.translations import get_text
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f"Python version: {sys.version}")
logger.debug(f"Python path: {sys.path}")

try:
    logger.debug("Attempting to import WordCloud")
    from wordcloud import WordCloud
    logger.debug("WordCloud imported successfully")
    
    logger.debug("Attempting to import matplotlib")
    import matplotlib.pyplot as plt
    logger.debug("matplotlib imported successfully")
    
    WORDCLOUD_AVAILABLE = True
    logger.info("WordCloud and matplotlib successfully imported")
except ImportError as e:
    WORDCLOUD_AVAILABLE = False
    logger.error(f"Error importing WordCloud or matplotlib: {e}", exc_info=True)

def display_query_result(result: Dict[str, Any], lang: str):
    st.subheader(get_text("query_response", lang))
    st.write(result['response'])

    if 'relevant_chunks' in result and result['relevant_chunks']:
        for i, chunk in enumerate(result['relevant_chunks'], 1):
            display_chunk(chunk, i, "chunk", lang)
    else:
        st.info(get_text("no_relevant_chunks", lang))

def display_search_results(results: List[Dict[str, Any]], lang: str):
    st.subheader(get_text("search_results", lang))
    if results:
        for i, result in enumerate(results, 1):
            display_chunk(result, i, "result", lang)
    else:
        st.info(get_text("no_search_results", lang))

def display_chunk(chunk: Dict[str, Any], index: int, lang: str):
    """Affiche un chunk sans expanders imbriqués"""
    st.subheader(f"Passage {index} (Score: {abs(chunk['similarity_score']):.4f})")
    st.markdown(chunk['chunk'])
    st.write("Métadonnées :")
    st.json(chunk['metadata'])
    st.divider()

def render_results(result_type: str, data: Any, lang: str):
    st.header(get_text("results_display", lang))
    
    logger.debug("=== BEGIN RENDER RESULTS ===")
    logger.debug(f"Result type: {result_type}")
    logger.debug(f"Data structure: {data.keys() if isinstance(data, dict) else 'Not a dict'}")

    if WORDCLOUD_AVAILABLE:
        try:
            if result_type == "query":
                # Vérifier si nous avons une réponse
                if isinstance(data, dict) and 'response' in data:
                    response_data = data['response']
                    
                    # S'assurer que response_data est un dictionnaire
                    if isinstance(response_data, dict):
                        chunks = response_data.get('relevant_chunks', [])
                        logger.debug(f"Found {len(chunks)} chunks")
                        
                        # Extraire le texte des chunks
                        text_parts = []
                        for chunk in chunks:
                            if isinstance(chunk, dict) and 'chunk' in chunk:
                                text_parts.append(chunk['chunk'])
                        
                        text = " ".join(text_parts)
                        
                        if text.strip():
                            # Générer le nuage de mots
                            wordcloud = WordCloud(
                                width=800, 
                                height=400, 
                                background_color='white',
                                min_word_length=3,
                                max_words=100
                            ).generate(text)

                            # Afficher le nuage de mots
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Afficher la réponse
                            st.subheader("Réponse")
                            st.write(response_data.get('response', ''))
                            
                            # Afficher les passages
                            st.subheader("Passages pertinents")
                            for i, chunk in enumerate(chunks, 1):
                                display_chunk(chunk, i, lang)

                            # Graphique des scores
                            scores_df = pd.DataFrame([
                                {
                                    "Passage": f"Passage {i+1}",
                                    "Score": abs(chunk['similarity_score'])
                                }
                                for i, chunk in enumerate(chunks)
                            ])
                            if not scores_df.empty:
                                st.subheader("Scores de similarité")
                                st.bar_chart(scores_df.set_index("Passage"))
                        else:
                            st.warning("Pas de texte trouvé dans les chunks")
                    else:
                        st.warning("Format de réponse invalide")
                else:
                    st.error("Format de données invalide")

        except Exception as e:
            logger.error(f"Error in visualization: {e}", exc_info=True)
            st.warning(f"Erreur lors de la visualisation: {str(e)}")

    # Export des résultats
    if st.button("Exporter les résultats"):
        try:
            if isinstance(data, dict) and 'response' in data and isinstance(data['response'], dict):
                chunks = data['response'].get('relevant_chunks', [])
                if chunks:
                    df = pd.DataFrame(chunks)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name="resultats.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Pas de données à exporter")
        except Exception as e:
            logger.error(f"Error in export functionality: {e}", exc_info=True)
            st.error("Erreur lors de l'export des résultats")

    logger.debug("=== END RENDER RESULTS ===")