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

def display_chunk(chunk: Dict[str, Any], index: int, score_label: str, lang: str):
    with st.expander(f"{get_text(score_label, lang)} {index} ({get_text('score', lang)}: {chunk['similarity_score']:.4f})"):
        st.write(chunk['chunk'])
        st.write(get_text("metadata", lang))
        st.json(chunk['metadata'])

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

def render_results(result_type: str, data: Any, lang: str):
    st.header(get_text("results_display", lang))

    if result_type == "query":
        display_query_result(data, lang)
    elif result_type == "search":
        display_search_results(data, lang)
    else:
        st.error(get_text("unknown_result_type", lang).format(result_type))

    if (result_type == "query" and 'relevant_chunks' in data and data['relevant_chunks']) or \
       (result_type == "search" and data):
        st.subheader(get_text("result_analysis", lang))

        if result_type == "query":
            df = pd.DataFrame([
                {"Chunk": f"{get_text('chunk', lang)} {i+1}", "Similarity Score": chunk['similarity_score']}
                for i, chunk in enumerate(data['relevant_chunks'])
            ])
        else: 
            df = pd.DataFrame([
                {"Result": f"{get_text('result', lang)} {i+1}", "Similarity Score": result['similarity_score']}
                for i, result in enumerate(data)
            ])

        st.bar_chart(df.set_index("Chunk" if result_type == "query" else "Result")["Similarity Score"])

        if WORDCLOUD_AVAILABLE:
            try:
                st.subheader(get_text("word_cloud", lang))
                text = data['response'] if result_type == "query" else ' '.join([r['chunk'] for r in data])
                logger.debug(f"Generating word cloud for text: {text[:100]}...")  # Log first 100 chars of text
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                logger.debug("Word cloud generated and displayed successfully")
            except Exception as e:
                logger.error(f"Error generating word cloud: {e}", exc_info=True)
                st.warning(get_text("wordcloud_generation_failed", lang))
        else:
            logger.warning("WordCloud or matplotlib not available")
            st.info(get_text("wordcloud_not_available", lang))
            st.markdown("""
            It seems that WordCloud or matplotlib is not available in the current environment.
            Please check the following:
            1. Ensure you're running the application in the correct virtual environment.
            2. Try reinstalling the libraries:
               ```
               pip uninstall wordcloud matplotlib
               pip install wordcloud matplotlib
               ```
            3. If the problem persists, check the application logs for more details.
            """)

    # Export results
    if st.button(get_text("export_results", lang)):
        if result_type == "query" and 'relevant_chunks' in data:
            df = pd.DataFrame(data['relevant_chunks'])
        elif result_type == "search":
            df = pd.DataFrame(data)
        else:
            st.warning(get_text("no_data_to_export", lang))
            return

        csv = df.to_csv(index=False)
        st.download_button(
            label=get_text("download_csv", lang),
            data=csv,
            file_name="results.csv",
            mime="text/csv",
        )

