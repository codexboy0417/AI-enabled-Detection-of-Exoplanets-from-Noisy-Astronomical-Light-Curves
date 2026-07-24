import os
import sys

def main():
    print("==================================================")
    print("Exoplanet Data Downloader (ISRO Hackathon Pipeline)")
    print("==================================================")
    
    try:
        import lightkurve as lk
        import pandas as pd
        import numpy as np
    except ImportError:
        print("[!] Missing dependencies. Please install them using:")
        print("    pip install lightkurve pandas astropy numpy")
        print("\n[i] Simulating fallback data download...")
        simulate_download()
        return

    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    target = "Kepler-186"
    print(f"[*] Querying MAST Archive for target: {target}...")
    
    try:
        # Search for Kepler light curves for the target
        search_result = lk.search_lightcurve(target, mission="Kepler", author="Kepler")
        print(search_result)
        
        if len(search_result) == 0:
            print("[!] No light curves found. Falling back to simulation.")
            simulate_download()
            return
            
        print("[*] Downloading the first available light curve quarter...")
        lc = search_result[0].download()
        
        # Clean up NaN values
        lc = lc.remove_nans()
        
        # Extract time and normalized flux
        time = lc.time.value
        flux = lc.flux.value
        
        # Normalize flux to average 1.0
        flux = flux / np.nanmedian(flux)
        
        df = pd.DataFrame({
            "time": time,
            "flux": flux
        })
        
        output_path = os.path.join("data", f"{target.lower().replace('-', '_')}_raw.csv")
        df.to_csv(output_path, index=False)
        print(f"[+] Download complete! Saved raw light curve to: {output_path}")
        print(f"    Total data points: {len(df)}")
        
    except Exception as e:
        print(f"[!] Error downloading data: {e}")
        print("[i] Falling back to simulation...")
        simulate_download()

def simulate_download():
    import numpy as np
    import pandas as pd
    
    os.makedirs("data", exist_ok=True)
    print("[*] Generating synthetic Kepler-186 light curve...")
    
    # Generate 1500 points (approx 30 days of observations at 30 min cadence)
    time = np.linspace(0, 30, 1500)
    
    # Base flux is 1.0
    flux = np.ones_like(time)
    
    # Exoplanet transit parameters
    period = 12.94  # days
    transit_depth = 0.008  # 0.8% dip
    duration = 0.15  # days (approx 3.6 hours)
    t0 = 4.0  # first transit time
    
    # Apply transits
    for t_transit in np.arange(t0, time[-1], period):
        phase = time - t_transit
        transit_mask = (phase >= -duration/2) & (phase <= duration/2)
        # Trapezoidal dip model
        flux[transit_mask] -= transit_depth * (1.0 - np.abs(phase[transit_mask]) / (duration/2) * 0.1)

    # Add stellar variability (low frequency sine wave)
    variability = 0.003 * np.sin(2 * np.pi * time / 5.0)
    flux += variability
    
    # Add random instrumental noise (high frequency white noise)
    noise = np.random.normal(0, 0.004, len(time))
    flux += noise
    
    # Add outliers (flares or cosmic rays)
    outlier_indices = np.random.choice(len(time), 10, replace=False)
    flux[outlier_indices] += np.random.uniform(0.015, 0.03, 10)
    
    df = pd.DataFrame({"time": time, "flux": flux})
    output_path = os.path.join("data", "kepler_186_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"[+] Synthetic data generation complete! Saved to: {output_path}")

if __name__ == "__main__":
    main()
