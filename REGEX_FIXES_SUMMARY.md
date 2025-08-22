# Regex Pattern Fixes Summary

## Issues Fixed

The regex patterns for ZIP file matching and field name detection were not working correctly with the file naming conventions. This document summarizes the fixes applied.

## Files Modified

### 1. `app/data/data_processing.py`

**Fixed Functions:**
- `_process_one_k()` - Improved ZIP-TIFF file matching
- `bulk_unzip_and_analyze_new_parallel()` - Enhanced field name detection
- `extract_date_from_filename()` - Better date extraction patterns

**Key Improvements:**
- More flexible ZIP file matching with pattern: `^{k_str}\.\s*.*\.zip$`
- Improved date extraction with Spanish month names: `(\d{1,2}(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\d{4})`
- Multiple fallback patterns for field name detection
- Better handling of various field name formats (underscores, spaces, double underscores)

### 2. `app/desktop_app/ui_desktop.py`

**Fixed Sections:**
- ZIP file upload field name detection
- Backup field name extraction during bulk analysis

**Key Improvements:**
- Updated to use the same improved patterns as data_processing.py
- Multiple pattern matching for better field name detection
- Consistent field name cleaning logic

## Pattern Examples

### Field Name Detection Patterns

1. **Pattern 1:** `\d+\.\s*([^_\s]+(?:[_\s][^_\s]+)*?)_+NDVI`
   - Matches: "001. Campo_Luna_Roja_NDVI_31ene2022.zip" → "Campo Luna Roja"

2. **Pattern 2:** `\d+\.\s*([^_]+(?:__[^_]+)*?)_+NDVI`
   - Matches: "001. perimetro__prev_NDVI_31ene2022.zip" → "Perimetro Prev"

3. **Pattern 3:** `\d+\.\s*(.+?)_NDVI`
   - Generic fallback for any format before NDVI

### Date Extraction Patterns

1. **Spanish Months:** `(\d{1,2}(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\d{4})`
   - Matches: "01ene2022", "15jul2023", "30dic2024"

2. **Generic 3-letter:** `(\d{1,2}[a-zA-Z]{3}\d{4})`
   - Fallback for other month abbreviations

3. **2-digit year:** `(\d{1,2}[a-zA-Z]{3}\d{2})`
   - Fallback for 2-digit years

### ZIP-TIFF Matching

- **ZIP Pattern:** `^{k_str}\.\s*.*\.zip$` where k_str is zero-padded (e.g., "001")
- **Date Matching:** Extract date from ZIP filename and match with TIFF files
- **ColorMap Detection:** Check for "colormap" keyword in filename (case-insensitive)

## Field Name Cleaning

The field name cleaning process:
1. Replace double underscores (`__`) with single underscores (`_`)
2. Replace underscores with spaces
3. Strip whitespace
4. Apply title case formatting

Example: `perimetro__prev` → `perimetro_prev` → `perimetro prev` → `Perimetro Prev`

## Testing

A test script `test_regex_patterns.py` was created to verify the patterns work correctly with various file naming conventions.

## Benefits

1. **Better Field Detection:** Handles various naming conventions automatically
2. **Improved ZIP Matching:** More reliable pairing of ZIP and TIFF files
3. **Robust Date Extraction:** Works with Spanish month abbreviations
4. **Consistent UI Updates:** Field names are properly detected and displayed
5. **Error Reduction:** Fewer failed matches due to naming variations

## Compatibility

The fixes maintain backward compatibility with existing file naming conventions while adding support for new formats.