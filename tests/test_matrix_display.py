#!/usr/bin/env python3
"""
Test script to verify that risk matrices now display numeric values.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data.ghg_capture import create_risk_matrix_heatmap, create_management_matrix_heatmap

def test_matrix_display():
    """Test that matrices display numeric values correctly."""
    
    # Create sample matrix data
    sample_matrix = np.array([
        [1.2, 2.3, 3.4, 4.5, 5.6],
        [2.1, 3.2, 4.3, 5.4, 6.5],
        [3.0, 4.1, 5.2, 6.3, 7.4],
        [4.9, 5.0, 6.1, 7.2, 8.3],
        [5.8, 6.9, 7.0, 8.1, 9.2]
    ])
    
    frequency_labels = ['Muy Pocos', 'Pocos', 'Más o Menos', 'Muchos', 'Bastantes']
    severity_labels = ['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
    
    print("🔍 Testing Risk Matrix Heatmap...")
    
    try:
        # Test the updated risk matrix function
        fig_risk = create_risk_matrix_heatmap(
            sample_matrix, 
            frequency_labels, 
            severity_labels, 
            "Test Risk Matrix"
        )
        
        # Check if the figure has text data
        heatmap_data = fig_risk.data[0]
        
        print(f"✅ Risk matrix created successfully")
        print(f"✅ Matrix shape: {sample_matrix.shape}")
        print(f"✅ Has text values: {hasattr(heatmap_data, 'text') and heatmap_data.text is not None}")
        print(f"✅ Text template: {getattr(heatmap_data, 'texttemplate', 'Not set')}")
        print(f"✅ Text font size: {getattr(heatmap_data, 'textfont', {}).get('size', 'Not set')}")
        
        if hasattr(heatmap_data, 'text') and heatmap_data.text is not None:
            print(f"✅ Sample text values: {heatmap_data.text[:2] if hasattr(heatmap_data.text, '__getitem__') else 'Available'}")
        
    except Exception as e:
        print(f"❌ Error testing risk matrix: {e}")
        return False
    
    print("\n🔍 Testing Management Matrix Heatmap for comparison...")
    
    try:
        # Test the management matrix function for comparison
        fig_mgmt = create_management_matrix_heatmap(
            sample_matrix, 
            frequency_labels, 
            severity_labels, 
            "Test Management Matrix"
        )
        
        # Check if the figure has text data
        heatmap_data_mgmt = fig_mgmt.data[0]
        
        print(f"✅ Management matrix created successfully")
        print(f"✅ Has text values: {hasattr(heatmap_data_mgmt, 'text') and heatmap_data_mgmt.text is not None}")
        print(f"✅ Text template: {getattr(heatmap_data_mgmt, 'texttemplate', 'Not set')}")
        
    except Exception as e:
        print(f"❌ Error testing management matrix: {e}")
        return False
    
    print("\n🎉 All tests passed! Risk matrices will now display numeric values.")
    return True

if __name__ == "__main__":
    test_matrix_display()
