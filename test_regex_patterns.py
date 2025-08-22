#!/usr/bin/env python3
"""
Test script to verify the improved regex patterns for ZIP file matching and field name detection.
"""

import re
import os

def test_field_name_detection():
    """Test the improved field name detection patterns."""
    
    # Test cases with various ZIP file naming conventions
    test_files = [
        "001. Campo_Luna_Roja_NDVI_01ene2022.zip",
        "002. Campo_Luna_Roja_NDVI_03ene2022.zip", 
        "001. perimetro__prev_NDVI_31ene2022.zip",
        "002. perimetro__prev_NDVI_12mar2022.zip",
        "003. Test_Field_Name_NDVI_15jul2023.zip",
        "004. Another Field NDVI_20ago2024.zip",
        "005. Simple_NDVI_25dic2024.zip"
    ]
    
    # Improved patterns from the updated code
    patterns = [
        # Pattern 1: "001. Campo_Luna_Roja_NDVI_31ene2022.zip"
        r'\d+\.\s*([^_\s]+(?:[_\s][^_\s]+)*?)_+NDVI',
        # Pattern 2: "001. perimetro__prev_NDVI_31ene2022.zip"
        r'\d+\.\s*([^_]+(?:__[^_]+)*?)_+NDVI',
        # Pattern 3: Generic field extraction before NDVI
        r'\d+\.\s*(.+?)_NDVI'
    ]
    
    print("Testing Field Name Detection:")
    print("=" * 50)
    
    for file in test_files:
        print(f"\nFile: {file}")
        detected_field = None
        
        for i, pattern in enumerate(patterns, 1):
            match = re.search(pattern, file, re.IGNORECASE)
            if match:
                raw_field = match.group(1)
                # Clean up the field name
                detected_field = raw_field.replace('__', '_').replace('_', ' ').strip().title()
                print(f"  Pattern {i} matched: '{raw_field}' -> '{detected_field}'")
                break
        
        if not detected_field:
            print(f"  No pattern matched!")

def test_date_extraction():
    """Test the improved date extraction patterns."""
    
    test_files = [
        "001. Campo_Luna_Roja_NDVI_01ene2022.zip",
        "002. Campo_Luna_Roja_NDVI_03feb2022.zip",
        "003. Campo_Luna_Roja_NDVI_16mar2022.zip",
        "004. Campo_Luna_Roja_NDVI_23abr2022.zip",
        "005. Campo_Luna_Roja_NDVI_26may2022.zip",
        "006. Campo_Luna_Roja_NDVI_28jun2022.zip",
        "007. Campo_Luna_Roja_NDVI_31jul2022.zip",
        "008. Campo_Luna_Roja_NDVI_10ago2022.zip",
        "009. Campo_Luna_Roja_NDVI_07sep2022.zip",
        "010. Campo_Luna_Roja_NDVI_14oct2022.zip",
        "011. Campo_Luna_Roja_NDVI_28nov2022.zip",
        "012. Campo_Luna_Roja_NDVI_01dic2022.zip"
    ]
    
    # Improved date patterns
    patterns = [
        r'(\d{1,2}(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\d{4})',  # Spanish months
        r'(\d{1,2}[a-zA-Z]{3}\d{4})',  # Generic 3-letter month
        r'(\d{1,2}[a-zA-Z]{3}\d{2})'   # 2-digit year fallback
    ]
    
    print("\n\nTesting Date Extraction:")
    print("=" * 50)
    
    for file in test_files:
        print(f"\nFile: {file}")
        extracted_date = None
        
        for i, pattern in enumerate(patterns, 1):
            match = re.search(pattern, file, re.IGNORECASE)
            if match:
                extracted_date = match.group(1)
                print(f"  Pattern {i} matched: '{extracted_date}'")
                break
        
        if not extracted_date:
            print(f"  No date pattern matched!")

def test_zip_matching():
    """Test the improved ZIP file matching logic."""
    
    # Simulate a folder with ZIP and TIFF files
    test_files = [
        "001. Campo_Luna_Roja_NDVI_01ene2022.zip",
        "001. Campo_Luna_Roja_NDVI_01ene2022.tiff",
        "001. Campo_Luna_Roja_NDVI_ColorMap_01ene2022.tiff",
        "002. Campo_Luna_Roja_NDVI_03ene2022.zip",
        "002. Campo_Luna_Roja_NDVI_03ene2022.tiff", 
        "002. Campo_Luna_Roja_NDVI_ColorMap_03ene2022.tiff"
    ]
    
    print("\n\nTesting ZIP-TIFF Matching:")
    print("=" * 50)
    
    zip_files = [f for f in test_files if f.lower().endswith('.zip')]
    tiff_files = [f for f in test_files if f.lower().endswith('.tiff')]
    
    for k_val in [1, 2]:
        k_str = str(k_val).zfill(3)
        print(f"\nTesting k_val={k_val} (k_str='{k_str}'):")
        
        # Find matching ZIP file
        matching_zip = None
        for zf in zip_files:
            zip_pattern = rf'^{k_str}\.\s*.*\.zip$'
            if re.match(zip_pattern, zf, re.IGNORECASE):
                matching_zip = zf
                print(f"  Found matching ZIP: {matching_zip}")
                break
        
        if matching_zip:
            # Extract date from ZIP filename
            date_match = re.search(r'(\d{1,2}(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\d{4})', matching_zip, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                print(f"  Extracted date: {date_str}")
                
                # Find corresponding TIFF files
                base_file = None
                color_file = None
                for fname in tiff_files:
                    if date_str.lower() in fname.lower():
                        if "colormap" in fname.lower():
                            color_file = fname
                            print(f"  Found color file: {color_file}")
                        else:
                            base_file = fname
                            print(f"  Found base file: {base_file}")
                
                if base_file and color_file:
                    print(f"  ✅ Complete pair found for k={k_val}")
                else:
                    print(f"  ❌ Incomplete pair for k={k_val}")
            else:
                print(f"  ❌ No date extracted from ZIP")
        else:
            print(f"  ❌ No matching ZIP found for k={k_val}")

if __name__ == "__main__":
    test_field_name_detection()
    test_date_extraction()
    test_zip_matching()
    print("\n" + "=" * 50)
    print("Testing completed!")