import os
import glob
import numpy as np
import pandas as pd

def main():
    print("==================================================")
    print("FITS Ingestion & Preprocessing Suite")
    print("==================================================")
    
    try:
        from astropy.io import fits
    except ImportError:
        print("[!] Missing Astropy. Please install it using:")
        print("    pip install astropy")
        return
        
    raw_dir = "data/raw_fits"
    fits_files = glob.glob(os.path.join(raw_dir, "*.fits"))
    
    total_files = len(fits_files)
    print(f"[*] Found {total_files} raw FITS files in {raw_dir}...")
    
    if total_files == 0:
        print("[!] No FITS files found to process. Please run download_bulk.py first.")
        return
        
    seq_len = 1000  # fixed length for neural network inputs
    
    processed_x = [] # noisy inputs
    processed_y = [] # clean targets
    processed_labels = [] # 1 for transit injected, 0 for stellar background
    
    print(f"[*] Ingesting and detrending FITS curves (target sequence length = {seq_len})...")
    
    count = 0
    for idx, filepath in enumerate(fits_files):
        try:
            with fits.open(filepath) as hdul:
                # Standard TESS/Kepler light curve files store data in binary tables in HDU index 1
                data = hdul[1].data
                
                # Check for columns
                cols = data.names
                time_col = 'TIME'
                # TESS uses PDCSAP_FLUX (Pre-search Data Conditioning SAP Flux) which has stellar trends removed
                flux_col = 'PDCSAP_FLUX' if 'PDCSAP_FLUX' in cols else 'SAP_FLUX'
                
                if time_col not in cols or flux_col not in cols:
                    continue
                    
                time = np.array(data[time_col])
                flux = np.array(data[flux_col])
                
                # Filter out NaNs
                nan_mask = ~np.isnan(time) & ~np.isnan(flux)
                time = time[nan_mask]
                flux = flux[nan_mask]
                
                if len(flux) < seq_len:
                    continue # Skip short files
                    
                # Take a fixed slice of the light curve
                flux = flux[:seq_len]
                time = time[:seq_len]
                
                # Normalize flux
                median_flux = np.median(flux)
                if median_flux == 0:
                    continue
                flux = flux / median_flux
                
                # Apply outlier clipping (positive anomalies)
                std_flux = np.std(flux)
                outlier_mask = flux > (1.0 + 3.0 * std_flux)
                flux[outlier_mask] = 1.0
                
                # Simple running median filter to detrend
                window = 51
                pad_size = window // 2
                padded = np.pad(flux, pad_size, mode='edge')
                trend = np.array([np.median(padded[i : i + window]) for i in range(len(flux))])
                detrended = flux / trend
                
                # Save base clean curve
                clean = detrended.copy()
                
                # Decide label & Augmentation (Inject synthetic transits)
                # 50% chance to inject an exoplanet transit
                label = 1 if idx % 2 == 0 else 0
                
                if label == 1:
                    # Injected Transit Parameters
                    depth = np.random.uniform(0.003, 0.015) # 0.3% to 1.5% dip
                    duration = np.random.randint(15, 50)
                    center = np.random.randint(200, 800)
                    
                    t_start = center - duration // 2
                    t_end = center + duration // 2
                    
                    for t_idx in range(t_start, t_end):
                        if 0 <= t_idx < seq_len:
                            # Simple trapezoid dip
                            dist = abs(t_idx - center)
                            if dist > (duration / 2) * 0.8:
                                # Smooth boundary
                                clean[t_idx] -= depth * ((duration/2 - dist) / (duration/2 * 0.2))
                            else:
                                clean[t_idx] -= depth
                                
                # Create noisy input from clean curve by adding high-frequency noise
                noise = np.random.normal(0, np.random.uniform(0.001, 0.004), seq_len)
                noisy = clean + noise
                
                processed_x.append(noisy)
                processed_y.append(clean)
                processed_labels.append(label)
                
                count += 1
                if count % 200 == 0:
                    print(f"    Processed {count}/{total_files} FITS files...")
                    
        except Exception as e:
            # Skip corrupted fits files
            continue
            
    if count == 0:
        print("[!] Failed to preprocess any files. Check that fits files are in data/raw_fits/")
        return
        
    print(f"[*] Packing preprocessed arrays...")
    X_arr = np.array(processed_x, dtype=np.float32)[:, np.newaxis, :] # (N, 1, seq_len)
    Y_arr = np.array(processed_y, dtype=np.float32)[:, np.newaxis, :] # (N, 1, seq_len)
    L_arr = np.array(processed_labels, dtype=np.float32)[:, np.newaxis] # (N, 1)
    
    # Save as highly compressed NPZ file
    output_path = "data/preprocessed_dataset.npz"
    np.savez_compressed(output_path, X=X_arr, Y=Y_arr, labels=L_arr)
    
    print(f"[+] Ingestion complete! Saved Packed Dataset to: {output_path}")
    print(f"    Total Packed Samples: {count}")
    print(f"    Input Array Shape:    {X_arr.shape}")
    print(f"    Target Array Shape:   {Y_arr.shape}")
    print(f"    Labels Array Shape:   {L_arr.shape}")
    print(f"    Total File Size:      {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
