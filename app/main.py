
import streamlit as st
from ui import render_ui
import config  # Ensure config.py is imported to initialize logging

def main():
    st.set_page_config(
        page_title="Crop Health Visualization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_ui()

if __name__ == "__main__":
    main()