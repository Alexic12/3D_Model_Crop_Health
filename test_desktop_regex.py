#!/usr/bin/env python3
"""
Test the desktop UI regex patterns to ensure they match the data processing patterns.
"""

import re

def test_desktop_field_detection():
    """Test field name detection patterns used in desktop UI."""
    
    test_files = [
        "001. Campo_Luna_Roja_NDVI_01ene2022.zip",
        "002. Campo_Luna_Roja_NDVI_03ene2022.zip", 
        "001. perimetro__prev_NDVI_31ene2022.zip",
        "002. perimetro__prev_NDVI_12mar2022.zip"
    ]
    
    print("Testing Desktop UI Field Detection:")
    print("=" * 50)
    
    for first_zip in test_files:
        print(f"\nFile: {first_zip}")
        detected_field = None
        
        # Desktop UI patterns (improved)
        patterns = [
            # Pattern 1: "001. Campo_Luna_Roja_NDVI_31ene2022.zip"
            r'\d+\.\s*([^_\s]+(?:[_\s][^_\s]+)*?)_+NDVI',
            # Pattern 2: "001. perimetro__prev_NDVI_31ene2022.zip"
            r'\d+\.\s*([^_]+(?:__[^_]+)*?)_+NDVI',
            # Pattern 3: Generic field extraction before NDVI
            r'\d+\.\s*(.+?)_NDVI'
        ]
        
        for i, pattern in enumerate(patterns, 1):
            match = re.search(pattern, first_zip, re.IGNORECASE)
            if match:
                raw_field = match.group(1)
                # Clean up the field name
                detected_field = raw_field.replace('__', '_').replace('_', ' ').strip().title()
                print(f"  Pattern {i} matched: '{raw_field}' -> '{detected_field}'")
                break
        
        if not detected_field:
            print(f"  ❌ No pattern matched!")
        else:
            print(f"  ✅ Final field name: '{detected_field}'")

if __name__ == "__main__":
    test_desktop_field_detection()