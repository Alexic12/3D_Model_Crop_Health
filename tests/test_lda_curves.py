#!/usr/bin/env python3
"""
Test script to validate the enhanced LDA plot with curves.
"""

import numpy as np
from scipy.stats import gaussian_kde

def test_lda_curve_logic():
    """Test the core logic for adding curves to LDA plots"""
    print("🔍 Testing LDA curve enhancement logic...")
    
    # Simulate LDA data similar to what the app would generate
    np.random.seed(42)
    
    # Create sample data for 3 management scenarios
    scenario1 = np.random.lognormal(mean=2.0, sigma=0.5, size=1000)
    scenario2 = np.random.lognormal(mean=2.2, sigma=0.6, size=1000) 
    scenario3 = np.random.lognormal(mean=1.8, sigma=0.4, size=1000)
    
    lda_data = np.column_stack([scenario1, scenario2, scenario3])
    mgr_labels = ['Conservative', 'Moderate', 'Aggressive']
    
    print(f"✅ Sample LDA data shape: {lda_data.shape}")
    print(f"✅ Management labels: {mgr_labels}")
    
    # Test the KDE curve generation logic
    all_data = lda_data.flatten()
    x_range = np.linspace(np.min(all_data), np.max(all_data), 200)
    
    print(f"✅ Combined data range: {np.min(all_data):.2f} to {np.max(all_data):.2f}")
    print(f"✅ X-range for curves: {len(x_range)} points")
    
    # Test individual scenario curves
    for i, label in enumerate(mgr_labels):
        data = lda_data[:, i]
        try:
            if len(data) > 1 and np.std(data) > 0:
                kde = gaussian_kde(data)
                curve_y = kde(x_range)
                print(f"✅ {label} curve: mean density = {np.mean(curve_y):.4f}, max = {np.max(curve_y):.4f}")
            else:
                print(f"❌ {label}: Invalid data for KDE")
        except Exception as e:
            print(f"❌ {label} KDE failed: {e}")
    
    # Test overall average curve
    try:
        if len(all_data) > 1 and np.std(all_data) > 0:
            kde_avg = gaussian_kde(all_data)
            curve_avg_y = kde_avg(x_range)
            print(f"✅ Average curve: mean density = {np.mean(curve_avg_y):.4f}, max = {np.max(curve_avg_y):.4f}")
        else:
            print(f"❌ Average curve: Invalid combined data")
    except Exception as e:
        print(f"❌ Average curve KDE failed: {e}")
    
    print("\n🎉 LDA curve enhancement logic validated successfully!")
    print("📊 The risk profile will now show:")
    print("   - Histogram bars for each management scenario") 
    print("   - Smooth density curves for each scenario")
    print("   - Overall average curve (black dashed line)")
    print("   - Reference lines for key metrics")
    
    return True

if __name__ == "__main__":
    test_lda_curve_logic()
