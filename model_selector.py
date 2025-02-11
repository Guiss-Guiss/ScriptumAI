import streamlit as st
import ollama
import logging
from typing import List, Dict, Optional
from language_utils import get_translation
import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["llama2", "mistral", "deepseek-coder"]
TIMEOUT_SECONDS = 5

def get_available_models() -> List[Dict]:
    """
    Fetch available models from Ollama with error handling and timeouts.
    Returns list of model dictionaries with name and details.
    """
    try:
        # Use httpx for better timeout control
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.get("http://127.0.0.1:11434/api/tags")
            if response.status_code == 200:
                return response.json().get('models', [])
            else:
                logger.error(f"Failed to fetch models: HTTP {response.status_code}")
                return []
    except httpx.TimeoutException:
        logger.error("Timeout while fetching models from Ollama")
        return []
    except Exception as e:
        logger.error(f"Error fetching models from Ollama: {str(e)}")
        return []

def format_model_name(model: Dict) -> str:
    """Format model name with additional details if available."""
    name = model.get('name', '')
    size = model.get('size', '')
    if size:
        size_gb = round(int(size) / 1e9, 1)  # Convert to GB
        return f"{name} ({size_gb}GB)"
    return name

def get_model_list(default_models: List[str] = DEFAULT_MODELS) -> List[str]:
    """Get list of available models with fallback to defaults."""
    available_models = get_available_models()
    
    if not available_models:
        logger.warning("No models found from Ollama, using default list")
        return default_models
    
    model_names = [format_model_name(model) for model in available_models]
    return model_names if model_names else default_models

def extract_model_name(formatted_name: str) -> str:
    """Extract base model name from formatted string."""
    return formatted_name.split(" (")[0]

def render_model_selector(
    default_model: str = "llama2",
    key: Optional[str] = None
) -> str:
    """
    Render the model selector dropdown with error handling.
    
    Args:
        default_model: Default model to select
        key: Optional unique key for the selectbox
        
    Returns:
        Selected model name (without formatting)
    """
    try:
        model_list = get_model_list()
        
        # Find the index of the default model
        default_index = 0
        for i, model in enumerate(model_list):
            if default_model in model:  # Check if default_model is part of formatted name
                default_index = i
                break
        
        # Add description in the help text
        help_text = (
            f"{get_translation('model_selection_help')} "
            f"{get_translation('available_models')}: {', '.join(model_list)}"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_model = st.selectbox(
                get_translation("select_model"),
                options=model_list,
                index=default_index,
                help=help_text,
                key=key
            )
        
        with col2:
            if st.button(get_translation("refresh_models"), key=f"refresh_{key}"):
                st.cache_data.clear()
                st.rerun()
        
        return extract_model_name(selected_model)
        
    except Exception as e:
        logger.error(f"Error rendering model selector: {str(e)}")
        st.error(get_translation("model_selector_error"))
        return default_model
