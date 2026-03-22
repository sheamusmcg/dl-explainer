import streamlit as st


def init_state():
    """Initialize session state with defaults. Each page is mostly self-contained,
    so we only track minimal shared state here."""
    defaults = {
        "initialized": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
