import streamlit as st


def next_step_button(page_path: str, label: str):
    """Consistent forward navigation at the bottom of a page."""
    st.divider()
    st.page_link(page_path, label=label, icon=":material/arrow_forward:")
