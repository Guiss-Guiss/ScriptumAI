import streamlit as st
import ollama
import logging
from typing import List
from language_utils import get_translation

logger = logging.getLogger(__name__)

def get_available_models() -> List[str]:
    """Fetch available models from Ollama."""
    try:
        models = ollama.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        logger.error(f"Error fetching models from Ollama: {str(e)}")
        return ["deepseek-r1:32b"]  # Fallback to default model

def render_model_selector(default_model: str = "deepseek-r1:32b") -> str:
    """Render the model selector dropdown and return the selected model."""
    try:
        available_models = get_available_models()
        
        # Find the index of the default model, defaulting to 0 if not found
        default_index = available_models.index(default_model) if default_model in available_models else 0
        
        selected_model = st.selectbox(
            get_translation("select_model"),
            options=available_models,
            index=default_index,
            help=get_translation("model_selection_help")
        )
        
        return selected_model
    except Exception as e:
        logger.error(f"Error rendering model selector: {str(e)}")
        return default_model