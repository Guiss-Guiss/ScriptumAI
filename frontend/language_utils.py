import streamlit as st
import json
import os

LANGUAGE_FILE = 'frontend/user_language.json'

def ensure_language_file():
    os.makedirs(os.path.dirname(LANGUAGE_FILE), exist_ok=True)

def load_user_language():
    if os.path.exists(LANGUAGE_FILE):
        with open(LANGUAGE_FILE, 'r') as f:
            return json.load(f).get('language', 'en')
    return 'en'

def save_user_language(language):
    ensure_language_file()
    with open(LANGUAGE_FILE, 'w') as f:
        json.dump({'language': language}, f)

def get_user_language():
    if 'user_language' not in st.session_state:
        st.session_state.user_language = load_user_language()
    return st.session_state.user_language

def set_user_language(language):
    st.session_state.user_language = language
    save_user_language(language)

def render_language_selector(get_text):
    languages = {
        'en': 'English',
        'fr': 'Français',
        'es': 'Español'
    }
    current_lang = get_user_language()
    selected_lang = st.selectbox(
        get_text('select_language', current_lang),
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(current_lang)
    )
    if selected_lang != current_lang:
        set_user_language(selected_lang)
        st.rerun()