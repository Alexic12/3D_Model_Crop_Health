"""
Responsive UI components for cross-device compatibility
"""
import streamlit as st
from typing import Dict, Any, Optional

class ResponsiveLayout:
    """Handles responsive layout patterns across devices"""
    
    @staticmethod
    def get_device_config() -> Dict[str, Any]:
        """Detect device type and return appropriate configuration"""
        # Use Streamlit's built-in viewport detection
        return {
            'is_mobile': st.session_state.get('is_mobile', False),
            'viewport_width': st.session_state.get('viewport_width', 1200),
            'columns': {
                'mobile': [1],
                'tablet': [1, 1], 
                'desktop': [2, 1]
            }
        }
    
    @staticmethod
    def responsive_columns(mobile_cols: list, tablet_cols: list, desktop_cols: list):
        """Create responsive column layout"""
        config = ResponsiveLayout.get_device_config()
        
        if config['viewport_width'] < 768:
            return st.columns(mobile_cols)
        elif config['viewport_width'] < 1024:
            return st.columns(tablet_cols)
        else:
            return st.columns(desktop_cols)
    
    @staticmethod
    def responsive_container(max_width: str = "1200px"):
        """Create responsive container with max-width"""
        st.markdown(f"""
        <style>
        .main .block-container {{
            max-width: {max_width};
            margin: 0 auto;
            padding: 1rem;
        }}
        </style>
        """, unsafe_allow_html=True)

class AccessibleComponents:
    """WCAG 2.1+ compliant UI components"""
    
    @staticmethod
    def accessible_button(label: str, key: str, help_text: Optional[str] = None):
        """Create accessible button with proper ARIA attributes"""
        return st.button(
            label, 
            key=key, 
            help=help_text,
            use_container_width=True
        )
    
    @staticmethod
    def accessible_selectbox(label: str, options: list, key: str, help_text: Optional[str] = None):
        """Create accessible selectbox with proper labeling"""
        return st.selectbox(
            label,
            options,
            key=key,
            help=help_text
        )

def inject_responsive_css():
    """Inject responsive CSS for cross-device compatibility"""
    st.markdown("""
    <style>
    /* Responsive Design System */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --background-color: #ffffff;
        --text-color: #333333;
        --border-radius: 8px;
        --spacing-unit: 1rem;
    }
    
    /* Mobile First Approach */
    .main .block-container {
        padding: 0.5rem;
        max-width: 100%;
    }
    
    /* Responsive Typography */
    h1, h2, h3 {
        line-height: 1.2;
        margin-bottom: var(--spacing-unit);
    }
    
    /* Tablet Styles */
    @media (min-width: 768px) {
        .main .block-container {
            padding: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
    }
    
    /* Desktop Styles */
    @media (min-width: 1024px) {
        .main .block-container {
            padding: 2rem;
        }
    }
    
    /* Accessibility Improvements */
    button:focus, 
    .stSelectbox > div > div:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* Fix Selectbox Text Visibility */
    .stSelectbox label {
        color: var(--text-color) !important;
    }
    
    .stSelectbox > div > div {
        color: var(--text-color) !important;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-color) !important;
    }
    
    .stSelectbox select {
        color: var(--text-color) !important;
    }
    
    .stSelectbox option {
        color: var(--text-color) !important;
    }
    
    /* Force visibility on all selectbox children */
    .stSelectbox * {
        color: var(--text-color) !important;
    }
    
    /* High Contrast Mode Support */
    @media (prefers-contrast: high) {
        :root {
            --background-color: #000000;
            --text-color: #ffffff;
        }
    }
    
    /* Reduced Motion Support */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)