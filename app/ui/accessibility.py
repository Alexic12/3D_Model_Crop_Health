"""
WCAG 2.1+ Accessibility compliance module
"""
import streamlit as st
from typing import Optional, List, Dict, Any

class AccessibilityManager:
    """Manages accessibility features and WCAG compliance"""
    
    @staticmethod
    def inject_accessibility_features():
        """Inject comprehensive accessibility features"""
        st.markdown("""
        <style>
        /* WCAG 2.1 AA Compliance Styles */
        
        /* Focus Management */
        *:focus {
            outline: 2px solid #005fcc;
            outline-offset: 2px;
        }
        
        /* Skip Links */
        .skip-link {
            position: absolute;
            top: -40px;
            left: 6px;
            background: #000;
            color: #fff;
            padding: 8px;
            text-decoration: none;
            z-index: 9999;
            border-radius: 4px;
        }
        
        .skip-link:focus {
            top: 6px;
        }
        
        /* Color Contrast Improvements */
        .stButton > button {
            background-color: #0066cc;
            color: #ffffff;
            border: 2px solid #0066cc;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background-color: #0052a3;
            border-color: #0052a3;
        }
        
        .stButton > button:focus {
            box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.3);
        }
        
        /* High Contrast Mode Support */
        @media (prefers-contrast: high) {
            .stButton > button {
                background-color: #000000;
                color: #ffffff;
                border: 3px solid #ffffff;
            }
            
            .stSelectbox > div > div {
                border: 2px solid #000000;
                background-color: #ffffff;
            }
        }
        
        /* Reduced Motion Support */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        
        /* Screen Reader Only Content */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        
        /* Improved Form Labels */
        .stTextInput label,
        .stSelectbox label,
        .stNumberInput label {
            font-weight: 600;
            color: #333333;
            margin-bottom: 4px;
        }
        
        /* Error States */
        .error-input {
            border: 2px solid #d73527 !important;
        }
        
        .error-message {
            color: #d73527;
            font-size: 0.875rem;
            margin-top: 4px;
        }
        
        /* Success States */
        .success-message {
            color: #2e7d32;
            font-size: 0.875rem;
            margin-top: 4px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_skip_links():
        """Add skip navigation links"""
        st.markdown("""
        <a href="#main-content" class="skip-link">Skip to main content</a>
        <a href="#sidebar" class="skip-link">Skip to navigation</a>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def accessible_heading(text: str, level: int = 1, id: Optional[str] = None):
        """Create accessible heading with proper hierarchy"""
        if level < 1 or level > 6:
            level = 1
        
        id_attr = f'id="{id}"' if id else ''
        st.markdown(f'<h{level} {id_attr}>{text}</h{level}>', unsafe_allow_html=True)
    
    @staticmethod
    def accessible_button(
        label: str, 
        key: str, 
        help_text: Optional[str] = None,
        disabled: bool = False,
        aria_label: Optional[str] = None
    ):
        """Create accessible button with ARIA attributes"""
        aria_label_attr = aria_label or label
        
        # Use Streamlit's built-in button with accessibility enhancements
        button_clicked = st.button(
            label,
            key=key,
            help=help_text,
            disabled=disabled,
            use_container_width=True
        )
        
        # Add ARIA label via JavaScript (in real implementation, would use custom component)
        if aria_label:
            st.markdown(f"""
            <script>
            document.querySelector('[data-testid="baseButton-{key}"]')?.setAttribute('aria-label', '{aria_label_attr}');
            </script>
            """, unsafe_allow_html=True)
        
        return button_clicked
    
    @staticmethod
    def accessible_form_field(
        field_type: str,
        label: str,
        key: str,
        required: bool = False,
        error_message: Optional[str] = None,
        help_text: Optional[str] = None,
        **kwargs
    ):
        """Create accessible form field with proper labeling and validation"""
        # Add required indicator
        label_text = f"{label} {'*' if required else ''}"
        
        # Create the field based on type
        if field_type == "text":
            value = st.text_input(label_text, key=key, help=help_text, **kwargs)
        elif field_type == "select":
            value = st.selectbox(label_text, key=key, help=help_text, **kwargs)
        elif field_type == "number":
            value = st.number_input(label_text, key=key, help=help_text, **kwargs)
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
        
        # Show error message if present
        if error_message:
            st.markdown(f'<div class="error-message" role="alert">{error_message}</div>', 
                       unsafe_allow_html=True)
        
        return value
    
    @staticmethod
    def announce_to_screen_reader(message: str):
        """Announce message to screen readers"""
        st.markdown(f"""
        <div aria-live="polite" aria-atomic="true" class="sr-only">
            {message}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_landmark_roles():
        """Add ARIA landmark roles to page sections"""
        st.markdown("""
        <script>
        // Add landmark roles (in real implementation, would be done server-side)
        document.querySelector('[data-testid="stSidebar"]')?.setAttribute('role', 'navigation');
        document.querySelector('[data-testid="stSidebar"]')?.setAttribute('aria-label', 'Main navigation');
        document.querySelector('.main')?.setAttribute('role', 'main');
        document.querySelector('.main')?.setAttribute('id', 'main-content');
        </script>
        """, unsafe_allow_html=True)

class KeyboardNavigation:
    """Handles keyboard navigation improvements"""
    
    @staticmethod
    def enhance_keyboard_navigation():
        """Add keyboard navigation enhancements"""
        st.markdown("""
        <script>
        // Enhanced keyboard navigation
        document.addEventListener('keydown', function(e) {
            // Escape key to close modals/dropdowns
            if (e.key === 'Escape') {
                // Close any open dropdowns
                document.querySelectorAll('.stSelectbox > div > div[aria-expanded="true"]').forEach(el => {
                    el.click();
                });
            }
            
            // Tab navigation improvements
            if (e.key === 'Tab') {
                // Ensure focus is visible
                document.body.classList.add('keyboard-navigation');
            }
        });
        
        // Remove keyboard navigation class on mouse use
        document.addEventListener('mousedown', function() {
            document.body.classList.remove('keyboard-navigation');
        });
        </script>
        
        <style>
        /* Only show focus outlines during keyboard navigation */
        body:not(.keyboard-navigation) *:focus {
            outline: none;
        }
        
        .keyboard-navigation *:focus {
            outline: 2px solid #005fcc;
            outline-offset: 2px;
        }
        </style>
        """, unsafe_allow_html=True)

def initialize_accessibility():
    """Initialize all accessibility features"""
    AccessibilityManager.inject_accessibility_features()
    AccessibilityManager.add_skip_links()
    AccessibilityManager.add_landmark_roles()
    KeyboardNavigation.enhance_keyboard_navigation()