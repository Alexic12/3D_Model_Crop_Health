"""
Mobile-specific optimizations and touch interactions
"""
import streamlit as st
from typing import Dict, Any

class MobileOptimizer:
    """Handles mobile-specific optimizations"""
    
    @staticmethod
    def detect_mobile() -> bool:
        """Detect if user is on mobile device"""
        # This would typically use JavaScript, but we'll use session state
        return st.session_state.get('is_mobile', False)
    
    @staticmethod
    def optimize_for_touch():
        """Apply touch-friendly optimizations"""
        st.markdown("""
        <style>
        /* Touch-friendly button sizing */
        .stButton > button {
            min-height: 44px;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        
        /* Touch-friendly form elements */
        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            min-height: 44px;
            font-size: 16px;
            padding: 12px;
        }
        
        /* Prevent zoom on input focus (iOS) */
        input, select, textarea {
            font-size: 16px !important;
        }
        
        /* Touch-friendly spacing */
        .element-container {
            margin-bottom: 1rem;
        }
        
        /* Swipe-friendly containers */
        .swipeable {
            touch-action: pan-x;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_mobile_navigation():
        """Add mobile-specific navigation patterns"""
        if MobileOptimizer.detect_mobile():
            st.markdown("""
            <style>
            /* Mobile navigation bar */
            .mobile-nav {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: white;
                border-top: 1px solid #ddd;
                padding: 8px;
                z-index: 1000;
                display: flex;
                justify-content: space-around;
            }
            
            .mobile-nav button {
                flex: 1;
                margin: 0 4px;
                padding: 12px 8px;
                border: none;
                background: transparent;
                font-size: 14px;
            }
            
            .mobile-nav button.active {
                background: #1f77b4;
                color: white;
                border-radius: 8px;
            }
            </style>
            """, unsafe_allow_html=True)

class PerformanceOptimizer:
    """Handles performance optimizations for mobile devices"""
    
    @staticmethod
    def lazy_load_components():
        """Implement lazy loading for heavy components"""
        if 'heavy_components_loaded' not in st.session_state:
            st.session_state.heavy_components_loaded = False
        
        if not st.session_state.heavy_components_loaded:
            if st.button("Load Visualization Components"):
                st.session_state.heavy_components_loaded = True
                st.rerun()
    
    @staticmethod
    def optimize_plotly_config() -> Dict[str, Any]:
        """Return mobile-optimized Plotly configuration"""
        return {
            'displayModeBar': False,  # Hide toolbar on mobile
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'crop_health_chart',
                'height': 500,
                'width': 700,
                'scale': 1
            },
            'modeBarButtonsToRemove': [
                'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian'
            ]
        }
    
    @staticmethod
    def reduce_data_for_mobile(data, max_points: int = 1000):
        """Reduce data points for mobile performance"""
        if len(data) > max_points:
            # Sample data to reduce load
            import numpy as np
            indices = np.linspace(0, len(data) - 1, max_points, dtype=int)
            return data.iloc[indices] if hasattr(data, 'iloc') else [data[i] for i in indices]
        return data

def inject_mobile_optimizations():
    """Inject all mobile optimizations"""
    MobileOptimizer.optimize_for_touch()
    MobileOptimizer.add_mobile_navigation()
    
    # Add JavaScript for device detection
    st.markdown("""
    <script>
    // Detect mobile device
    function isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }
    
    // Set viewport width
    function setViewportWidth() {
        return window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
    }
    
    // Update Streamlit session state (would need custom component in real implementation)
    if (typeof window.streamlitSetComponentValue !== 'undefined') {
        window.streamlitSetComponentValue('is_mobile', isMobile());
        window.streamlitSetComponentValue('viewport_width', setViewportWidth());
    }
    </script>
    """, unsafe_allow_html=True)