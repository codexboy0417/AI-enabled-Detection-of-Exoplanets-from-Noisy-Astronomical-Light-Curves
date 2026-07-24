import numpy as np
import random

SCENARIOS = {
    "kepler_186f": {
        "id": "kepler_186f",
        "name": "Kepler-186f (Habitable Zone Exoplanet)",
        "type": "Confirmed Exoplanet",
        "description": "An Earth-sized planet orbiting in the habitable zone of a cool M-dwarf star. Transit signal is tiny (~0.08% dip) and heavily obscured by noise.",
        "metadata": {
            "status": "Transit Detected",
            "confidence": 0.98,
            "period": 12.94,
            "depth": 0.08,
            "duration": 3.46,
            "radius": 1.17,
            "habitability": "Habitable Zone (Liquid Water Possible)",
            "starType": "M-Dwarf (Red Dwarf)",
            "distance": 582
        }
    },
    "kepler_90i": {
        "id": "kepler_90i",
        "name": "Kepler-90i (Multi-Planet System)",
        "type": "Confirmed Exoplanet",
        "description": "A super-Earth orbiting Kepler-90, the first star known to host 8 planets (like our solar system). Exhibits overlapping transit signatures.",
        "metadata": {
            "status": "Transit Detected (Multi-planet)",
            "confidence": 0.95,
            "period": 14.45,
            "depth": 0.15,
            "duration": 2.80,
            "radius": 1.32,
            "habitability": "Too Hot (Inferno Planet)",
            "starType": "G-type Star (Sun-like)",
            "distance": 2840
        }
    },
    "eclipsing_binary": {
        "id": "eclipsing_binary",
        "name": "Eclipsing Binary Star (False Positive)",
        "type": "False Positive",
        "description": "Two stars orbiting each other. The deep V-shaped dips look like transits but are actually another star blocking light. Easily misidentified without high-res models.",
        "metadata": {
            "status": "False Positive (Binary System)",
            "confidence": 0.02,
            "period": 8.52,
            "depth": 6.20,
            "duration": 5.12,
            "radius": 9.84,
            "habitability": "Not Applicable (Stellar Companion)",
            "starType": "Binary (K-dwarf + M-dwarf)",
            "distance": 920
        }
    },
    "stellar_activity": {
        "id": "stellar_activity",
        "name": "Active Starspots (False Positive)",
        "type": "False Positive",
        "description": "Stellar rotation carries large, dark starspots across the stellar disk, creating smooth sinusoidal variability that mimics transits.",
        "metadata": {
            "status": "False Positive (Stellar Noise)",
            "confidence": 0.05,
            "period": 5.20,
            "depth": 0.40,
            "duration": 24.0,
            "radius": 0.0,
            "habitability": "Not Applicable (Starspots)",
            "starType": "Active Solar-type G-star",
            "distance": 430
        }
    },
    "undiscovered_noisy": {
        "id": "undiscovered_noisy",
        "name": "TOI-2026 (Unexplored Noisy Star)",
        "type": "Candidate Exoplanet",
        "description": "A high-priority target with severe instrumental noise and high stellar activity. Denoising exposes a potential habitable super-Earth.",
        "metadata": {
            "status": "Transit Detected (Candidate)",
            "confidence": 0.89,
            "period": 19.82,
            "depth": 0.12,
            "duration": 4.10,
            "radius": 1.64,
            "habitability": "Habitable Zone (Super-Earth)",
            "starType": "K-dwarf (Orange Dwarf)",
            "distance": 145
        }
    }
}

def apply_transit(flux, time, period, depth, duration, t0):
    for idx, t in enumerate(time):
        transit_num = round((t - t0) / period)
        closest_transit = t0 + transit_num * period
        phase = t - closest_transit
        if abs(phase) <= duration / 2:
            border = duration * 0.12
            dist = abs(phase)
            if dist > duration / 2 - border:
                fraction = (duration / 2 - dist) / border
                flux[idx] -= depth * fraction
            else:
                flux[idx] -= depth

def apply_v_transit(flux, time, period, depth, duration, t0, is_primary):
    for idx, t in enumerate(time):
        transit_num = round((t - t0) / period)
        closest_transit = t0 + transit_num * period
        phase = t - closest_transit
        if abs(phase) <= duration / 2:
            ratio = abs(phase) / (duration / 2)
            flux[idx] -= depth * (1.0 - ratio)

def generate_light_curve(scenario_id, noise_scale=1.0):
    scenario = SCENARIOS.get(scenario_id)
    if not scenario:
        return None
    
    meta = scenario["metadata"]
    seq_len = 300
    time = np.array([i * 0.1 for i in range(seq_len)])
    
    clean_flux = np.ones(seq_len)
    trend_flux = np.ones(seq_len)
    
    # 1. Generate Clean Physical Model
    if scenario_id == "kepler_186f":
        apply_transit(clean_flux, time, meta["period"], meta["depth"] / 100, meta["duration"] / 24, 4.2)
    elif scenario_id == "kepler_90i":
        apply_transit(clean_flux, time, meta["period"], meta["depth"] / 100, meta["duration"] / 24, 3.1)
        apply_transit(clean_flux, time, 7.0, 0.0009, 2.0 / 24, 1.5)
    elif scenario_id == "eclipsing_binary":
        apply_v_transit(clean_flux, time, meta["period"], meta["depth"] / 100, meta["duration"] / 24, 2.5, True)
        apply_v_transit(clean_flux, time, meta["period"], 0.008, 3.5 / 24, 2.5 + meta["period"] / 2, False)
    elif scenario_id == "stellar_activity":
        for idx in range(seq_len):
            clean_flux[idx] += (meta["depth"] / 100) * np.sin((2 * np.pi * time[idx]) / meta["period"])
    elif scenario_id == "undiscovered_noisy":
        apply_transit(clean_flux, time, meta["period"], meta["depth"] / 100, meta["duration"] / 24, 6.5)

    # 2. Generate Low-Frequency Stellar Trend
    trend_amp = 0.0025
    trend_period = 12.0
    if scenario_id == "stellar_activity":
        trend_amp = 0.006
        trend_period = 8.0
    elif scenario_id == "undiscovered_noisy":
        trend_amp = 0.004
        trend_period = 15.0
        
    for idx in range(seq_len):
        trend_flux[idx] = 1.0 + trend_amp * np.sin((2 * np.pi * time[idx]) / trend_period)
        
    # 3. Combine Trend + Clean Model + Noise -> Raw
    white_noise_std = 0.002
    if scenario_id == "kepler_186f":
        white_noise_std = 0.0018
    elif scenario_id == "undiscovered_noisy":
        white_noise_std = 0.0045
        
    white_noise_std *= noise_scale
    raw_flux = np.zeros(seq_len)
    
    for idx in range(seq_len):
        white_noise = np.random.normal(0, white_noise_std)
        raw_flux[idx] = clean_flux[idx] * trend_flux[idx] + white_noise
        
        # Inject cosmic rays
        if random.random() < 0.01:
            raw_flux[idx] += random.uniform(0.01, 0.025)
            
    # 4. Simulate Denoised Autoencoder (DAE) Output
    filter_noise_std = 0.0003
    denoised_flux = np.zeros(seq_len)
    for idx in range(seq_len):
        denoised_flux[idx] = clean_flux[idx] + np.random.normal(0, filter_noise_std)
        
    # 5. Simulate Classifier Rolling Probability
    probability = np.zeros(seq_len)
    for idx in range(seq_len):
        dip_factor = 1.0 - clean_flux[idx]
        if dip_factor > 0.0001:
            if scenario_id == "eclipsing_binary":
                probability[idx] = 0.02 + dip_factor * 0.05
            elif scenario_id == "stellar_activity":
                probability[idx] = 0.03 + random.random() * 0.02
            else:
                probability[idx] = min(0.99, 0.85 + dip_factor * 120 + random.random() * 0.05)
        else:
            probability[idx] = max(0.005, 0.01 + np.random.normal(0, 0.005))
            
    # 6. Generate Phase-Folded Data
    phase_data = []
    P = meta["period"]
    t0 = 4.2
    if scenario_id == "kepler_90i":
        t0 = 3.1
    elif scenario_id == "eclipsing_binary":
        t0 = 2.5
    elif scenario_id == "undiscovered_noisy":
        t0 = 6.5
        
    for idx in range(seq_len):
        t = time[idx]
        phase = ((t - t0) / P) % 1.0
        if phase > 0.5:
            phase -= 1.0
        if phase < -0.5:
            phase += 1.0
            
        raw_detrended = raw_flux[idx] / trend_flux[idx]
        
        model_folded_val = 1.0
        if scenario_id in ["kepler_186f", "undiscovered_noisy", "kepler_90i"]:
            dur_fraction = (meta["duration"] / 24) / P
            if abs(phase) < dur_fraction / 2:
                model_folded_val = 1.0 - (meta["depth"] / 100)
        elif scenario_id == "eclipsing_binary":
            dur_fraction = (meta["duration"] / 24) / P
            if abs(phase) < dur_fraction / 2:
                ratio = abs(phase) / (dur_fraction / 2)
                model_folded_val = 1.0 - (meta["depth"] / 100) * (1.0 - ratio)
                
        phase_data.append({
            "phase": float(phase),
            "rawFlux": float(raw_detrended),
            "modelFlux": float(model_folded_val)
        })
        
    phase_data.sort(key=lambda x: x["phase"])
    
    raw_chart_data = []
    denoised_chart_data = []
    probability_chart_data = []
    
    for i in range(seq_len):
        raw_chart_data.append({
            "time": float(round(time[i], 2)),
            "rawFlux": float(round(raw_flux[i], 5)),
            "trend": float(round(trend_flux[i], 5))
        })
        denoised_chart_data.append({
            "time": float(round(time[i], 2)),
            "rawFlux": float(round(raw_flux[i]/trend_flux[i], 5)),
            "denoisedFlux": float(round(denoised_flux[i], 5)),
            "cleanFlux": float(round(clean_flux[i], 5))
        })
        probability_chart_data.append({
            "time": float(round(time[i], 2)),
            "probability": float(round(probability[i], 4)),
            "threshold": 0.5
        })
        
    return {
        "rawChartData": raw_chart_data,
        "denoisedChartData": denoised_chart_data,
        "probabilityChartData": probability_chart_data,
        "phaseFoldedData": phase_data,
        "metadata": meta
    }
