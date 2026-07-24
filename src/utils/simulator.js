// Exoplanet Transit and Stellar Noise Physics Simulation Engine
// Generates realistic light curves for the ISRO hackathon prototype

export const SCENARIOS = {
  kepler_186f: {
    id: "kepler_186f",
    name: "Kepler-186f (Habitable Zone Exoplanet)",
    type: "Confirmed Exoplanet",
    description: "An Earth-sized planet orbiting in the habitable zone of a cool M-dwarf star. Transit signal is tiny (~0.08% dip) and heavily obscured by noise.",
    metadata: {
      status: "Transit Detected",
      confidence: 0.98,
      period: 12.94, // days
      depth: 0.08,  // %
      duration: 3.46, // hours
      radius: 1.17, // Earth radii
      habitability: "Habitable Zone (Liquid Water Possible)",
      starType: "M-Dwarf (Red Dwarf)",
      distance: 582 // light years
    }
  },
  kepler_90i: {
    id: "kepler_90i",
    name: "Kepler-90i (Multi-Planet System)",
    type: "Confirmed Exoplanet",
    description: "A super-Earth orbiting Kepler-90, the first star known to host 8 planets (like our solar system). Exhibits overlapping transit signatures.",
    metadata: {
      status: "Transit Detected (Multi-planet)",
      confidence: 0.95,
      period: 14.45,
      depth: 0.15,
      duration: 2.80,
      radius: 1.32,
      habitability: "Too Hot (Inferno Planet)",
      starType: "G-type Star (Sun-like)",
      distance: 2840
    }
  },
  eclipsing_binary: {
    id: "eclipsing_binary",
    name: "Eclipsing Binary Star (False Positive)",
    type: "False Positive",
    description: "Two stars orbiting each other. The deep V-shaped dips look like transits but are actually another star block. Easily misidentified without high-res models.",
    metadata: {
      status: "False Positive (Binary System)",
      confidence: 0.02,
      period: 8.52,
      depth: 6.20,
      duration: 5.12,
      radius: 9.84, // Companion is stellar size
      habitability: "Not Applicable (Stellar Companion)",
      starType: "Binary (K-dwarf + M-dwarf)",
      distance: 920
    }
  },
  stellar_activity: {
    id: "stellar_activity",
    name: "Active Starspots (False Positive)",
    type: "False Positive",
    description: "Stellar rotation carries large, dark starspots across the stellar disk, creating smooth sinusoidal variability that mimics transits.",
    metadata: {
      status: "False Positive (Stellar Noise)",
      confidence: 0.05,
      period: 5.20, // Stellar rotation period
      depth: 0.40,
      duration: 24.0, // Long and gradual dip
      radius: 0.0,
      habitability: "Not Applicable (Starspots)",
      starType: "Active Solar-type G-star",
      distance: 430
    }
  },
  undiscovered_noisy: {
    id: "undiscovered_noisy",
    name: "TOI-2026 (Unexplored Noisy Star)",
    type: "Candidate Exoplanet",
    description: "A high-priority target with severe instrumental noise and high stellar activity. Denoising exposes a potential habitable super-Earth.",
    metadata: {
      status: "Transit Detected (Candidate)",
      confidence: 0.89,
      period: 19.82,
      depth: 0.12,
      duration: 4.10,
      radius: 1.64,
      habitability: "Habitable Zone (Super-Earth)",
      starType: "K-dwarf (Orange Dwarf)",
      distance: 145
    }
  }
};

// Generates time-series light curve data based on physics models
export function generateLightCurve(scenarioId, noiseScale = 1.0) {
  const scenario = SCENARIOS[scenarioId];
  if (!scenario) return null;
  
  const seqLen = 300;
  const time = Array.from({ length: seqLen }, (_, i) => i * 0.1); // 0 to 30 days
  
  let cleanFlux = Array(seqLen).fill(1.0);
  let trendFlux = Array(seqLen).fill(1.0);
  let rawFlux = Array(seqLen).fill(1.0);
  let denoisedFlux = Array(seqLen).fill(1.0);
  let probability = Array(seqLen).fill(0.01);
  
  const meta = scenario.metadata;
  
  // 1. Generate Clean Physical Model
  if (scenarioId === "kepler_186f") {
    applyTransit(cleanFlux, time, meta.period, meta.depth / 100, meta.duration / 24, 4.2);
  } else if (scenarioId === "kepler_90i") {
    // Primary planet (90i)
    applyTransit(cleanFlux, time, meta.period, meta.depth / 100, meta.duration / 24, 3.1);
    // Inner planet (90b) with period 7.0 days, depth 0.09%, duration 2.0h
    applyTransit(cleanFlux, time, 7.0, 0.0009, 2.0 / 24, 1.5);
  } else if (scenarioId === "eclipsing_binary") {
    // Sharp V-shaped primary eclipses (deep)
    applyVTransit(cleanFlux, time, meta.period, meta.depth / 100, meta.duration / 24, 2.5, true);
    // Secondary eclipses (shallow, 180 degrees out of phase)
    applyVTransit(cleanFlux, time, meta.period, 0.008, 3.5 / 24, 2.5 + meta.period / 2, false);
  } else if (scenarioId === "stellar_activity") {
    // Sinusoidal variability representing rotating starspots
    for (let idx = 0; idx < seqLen; idx++) {
      cleanFlux[idx] += (meta.depth / 100) * Math.sin((2 * Math.PI * time[idx]) / meta.period);
    }
  } else if (scenarioId === "undiscovered_noisy") {
    applyTransit(cleanFlux, time, meta.period, meta.depth / 100, meta.duration / 24, 6.5);
  }
  
  // 2. Generate Low-Frequency Stellar Trend (e.g. stellar rotation or breathing)
  let trendAmp = 0.0025;
  let trendPeriod = 12.0;
  if (scenarioId === "stellar_activity") {
    trendAmp = 0.006;
    trendPeriod = 8.0;
  } else if (scenarioId === "undiscovered_noisy") {
    trendAmp = 0.004;
    trendPeriod = 15.0;
  }
  
  for (let idx = 0; idx < seqLen; idx++) {
    trendFlux[idx] = 1.0 + trendAmp * Math.sin((2 * Math.PI * time[idx]) / trendPeriod);
  }
  
  // 3. Combine Trend + Clean Model + High Frequency Noise -> Raw Curve
  let whiteNoiseStd = 0.002; // standard noise level
  if (scenarioId === "kepler_186f") whiteNoiseStd = 0.0018;
  if (scenarioId === "undiscovered_noisy") whiteNoiseStd = 0.0045; // High noise
  
  whiteNoiseStd *= noiseScale;
  
  for (let idx = 0; idx < seqLen; idx++) {
    const whiteNoise = randomNormal(0, whiteNoiseStd);
    // Flux is multiplicative with trend
    rawFlux[idx] = cleanFlux[idx] * trendFlux[idx] + whiteNoise;
    
    // Inject cosmic ray outliers (random spikes)
    if (Math.random() < 0.01) {
      rawFlux[idx] += randomRange(0.01, 0.025);
    }
  }
  
  // 4. Simulate Denoised Autoencoder (DAE) Output
  // In reality, the DAE removes the high-frequency white noise and detrends the data.
  // We model this by applying a slight moving average/filter to the cleanFlux
  let filterNoiseStd = 0.0003; // small residual noise
  for (let idx = 0; idx < seqLen; idx++) {
    // The DAE removes the trend completely, keeping baseline at 1.0
    denoisedFlux[idx] = cleanFlux[idx] + randomNormal(0, filterNoiseStd);
  }
  
  // 5. Simulate Classifier Rolling Probability Curve
  // Probability spikes when the sliding window detects a transit shape in the denoised flux
  for (let idx = 0; idx < seqLen; idx++) {
    let dipFactor = 1.0 - cleanFlux[idx];
    if (dipFactor > 0.0001) {
      if (scenarioId === "eclipsing_binary") {
        // Deep binary eclipses show low exoplanet probability (vetoed by classifier)
        probability[idx] = 0.02 + dipFactor * 0.05;
      } else if (scenarioId === "stellar_activity") {
        // Sinusoidal stellar activity has low transit probability
        probability[idx] = 0.03 + Math.random() * 0.02;
      } else {
        // Exoplanet transit spikes probability
        probability[idx] = Math.min(0.99, 0.85 + dipFactor * 120 + Math.random() * 0.05);
      }
    } else {
      probability[idx] = Math.max(0.005, 0.01 + randomNormal(0, 0.005));
    }
  }
  
  // 6. Generate Phase-Folded Data
  const phaseData = [];
  const P = meta.period;
  // Fold time around the primary transit epoch t0
  let t0 = 4.2;
  if (scenarioId === "kepler_90i") t0 = 3.1;
  if (scenarioId === "eclipsing_binary") t0 = 2.5;
  if (scenarioId === "undiscovered_noisy") t0 = 6.5;
  
  for (let idx = 0; idx < seqLen; idx++) {
    const t = time[idx];
    // Calculate phase from -0.5 to 0.5
    let phase = ((t - t0) / P) % 1.0;
    if (phase > 0.5) phase -= 1.0;
    if (phase < -0.5) phase += 1.0;
    
    // Add raw (but detrended) folded flux and clean model folded flux
    const rawDetrended = rawFlux[idx] / trendFlux[idx];
    
    // Eclipsing binaries or stellar activity have different folding structures
    let modelFoldedVal = 1.0;
    if (scenarioId === "kepler_186f" || scenarioId === "undiscovered_noisy" || scenarioId === "kepler_90i") {
      const durFraction = (meta.duration / 24) / P;
      if (Math.abs(phase) < durFraction / 2) {
        modelFoldedVal = 1.0 - (meta.depth / 100);
      }
    } else if (scenarioId === "eclipsing_binary") {
      const durFraction = (meta.duration / 24) / P;
      if (Math.abs(phase) < durFraction / 2) {
        // V-shape model fit
        const ratio = Math.abs(phase) / (durFraction / 2);
        modelFoldedVal = 1.0 - (meta.depth / 100) * (1.0 - ratio);
      }
    }
    
    phaseData.push({
      phase,
      rawFlux: rawDetrended,
      modelFlux: modelFoldedVal
    });
  }
  
  // Sort phaseData by phase for plotting lines correctly
  phaseData.sort((a, b) => a.phase - b.phase);
  
  // Form datasets for charts
  const rawChartData = time.map((t, i) => ({
    time: parseFloat(t.toFixed(2)),
    rawFlux: parseFloat(rawFlux[i].toFixed(5)),
    trend: parseFloat(trendFlux[i].toFixed(5))
  }));
  
  const denoisedChartData = time.map((t, i) => ({
    time: parseFloat(t.toFixed(2)),
    rawFlux: parseFloat((rawFlux[i]/trendFlux[i]).toFixed(5)), // detrended raw
    denoisedFlux: parseFloat(denoisedFlux[i].toFixed(5)),
    cleanFlux: parseFloat(cleanFlux[i].toFixed(5))
  }));
  
  const probabilityChartData = time.map((t, i) => ({
    time: parseFloat(t.toFixed(2)),
    probability: parseFloat(probability[i].toFixed(4)),
    threshold: 0.5
  }));
  
  return {
    rawChartData,
    denoisedChartData,
    probabilityChartData,
    phaseFoldedData: phaseData,
    metadata: meta
  };
}

// Helpers
function randomNormal(mean = 0, std = 1) {
  // Box-Muller transform
  const u1 = 1.0 - Math.random();
  const u2 = 1.0 - Math.random();
  const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
  return mean + std * randStdNormal;
}

function randomRange(min, max) {
  return Math.random() * (max - min) + min;
}

function applyTransit(flux, time, period, depth, duration, t0) {
  const seqLen = flux.length;
  for (let idx = 0; idx < seqLen; idx++) {
    const t = time[idx];
    
    // Check if time is within any periodic transit window
    // Find closest transit epoch
    const transitNum = Math.round((t - t0) / period);
    const closestTransit = t0 + transitNum * period;
    
    const phase = t - closestTransit;
    if (Math.abs(phase) <= duration / 2) {
      // U-shape flat-bottomed dip (limb darkening modeled as soft trapezoid)
      const border = duration * 0.12;
      const dist = Math.abs(phase);
      if (dist > duration / 2 - border) {
        // Soft edge transition
        const fraction = (duration / 2 - dist) / border;
        flux[idx] -= depth * fraction;
      } else {
        // Flat bottom
        flux[idx] -= depth;
      }
    }
  }
}

function applyVTransit(flux, time, period, depth, duration, t0, isPrimary) {
  const seqLen = flux.length;
  for (let idx = 0; idx < seqLen; idx++) {
    const t = time[idx];
    const transitNum = Math.round((t - t0) / period);
    const closestTransit = t0 + transitNum * period;
    
    const phase = t - closestTransit;
    if (Math.abs(phase) <= duration / 2) {
      // Sharp V-shape (linear dip to bottom)
      const ratio = Math.abs(phase) / (duration / 2);
      flux[idx] -= depth * (1.0 - ratio);
    }
  }
}
