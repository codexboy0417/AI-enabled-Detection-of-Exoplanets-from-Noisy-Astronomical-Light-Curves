import os
import pandas as pd
import numpy as np

def remove_outliers_sigma_clipping(flux, sigma=3.0):
    """
    Remove outliers (e.g. stellar flares or instrument errors) using sigma clipping.
    """
    median = np.median(flux)
    std = np.std(flux)
    cleaned_flux = flux.copy()
    
    # Identify indices that exceed the threshold (only upper outliers since transits are dips)
    outliers = flux > (median + sigma * std)
    cleaned_flux[outliers] = median
    return cleaned_flux

def median_filter_detrend(flux, window_size=51):
    """
    Apply a running median filter to detrend the light curve (remove stellar variability).
    """
    # Pad flux at borders
    pad_size = window_size // 2
    padded_flux = np.pad(flux, pad_size, mode='edge')
    
    # Apply moving median
    trend = np.zeros_like(flux)
    for i in range(len(flux)):
        trend[i] = np.median(padded_flux[i : i + window_size])
        
    # Detrend by dividing by the trend (since flux is multiplicative)
    detrended = flux / trend
    return detrended, trend

def main():
    print("==================================================")
    print("Light Curve Preprocessing Pipeline")
    print("==================================================")
    
    input_file = "data/kepler_186_raw.csv"
    if not os.path.exists(input_file):
        # Check if alternative exists
        alt_file = "data/kepler_186_raw.csv"
        if not os.path.exists(alt_file):
            print(f"[!] Input file {input_file} not found. Please run download_data.py first.")
            return
        input_file = alt_file
        
    print(f"[*] Loading raw light curve from: {input_file}")
    df = pd.read_csv(input_file)
    
    time = df["time"].values
    flux = df["flux"].values
    
    # Step 1: Remove positive outliers (flares)
    print("[*] Applying 3-sigma clipping for positive outliers...")
    flux_no_outliers = remove_outliers_sigma_clipping(flux, sigma=3.0)
    
    # Step 2: Detrend using running median filter
    print("[*] Running median filter detrending (window size = 51)...")
    flux_detrended, trend = median_filter_detrend(flux_no_outliers, window_size=51)
    
    # Step 3: Save preprocessed data
    df_processed = pd.DataFrame({
        "time": time,
        "raw_flux": flux,
        "trend": trend,
        "processed_flux": flux_detrended
    })
    
    output_path = "data/kepler_186_processed.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"[+] Preprocessing complete! Saved to: {output_path}")
    print(f"    Raw Mean/Std: {np.mean(flux):.5f}/{np.std(flux):.5f}")
    print(f"    Cleaned Mean/Std: {np.mean(flux_detrended):.5f}/{np.std(flux_detrended):.5f}")

if __name__ == "__main__":
    main()
