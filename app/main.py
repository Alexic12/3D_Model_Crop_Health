import streamlit as st
from app.ui_wk import render_ui
import config  # Ensures logging is set up

def main():
    st.set_page_config(
        page_title="Crop Health Visualization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_ui()

if __name__ == "__main__":
    main()
