import React, { useState } from "react";
import { FileText, Copy, Check, Download, Layers } from "lucide-react";

export default function ProposalViewer() {
  const [copied, setCopied] = useState(false);

  const proposalText = `# BHARATIYA ANTARIKSH HACKATHON 2026
## TECHNICAL PROPOSAL & IDEA SUBMISSION

### PROBLEM STATEMENT: 
AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

---

### SLIDE 1: Title & Overview
*   **Project Title:** AstroPulse: Deep Denoising & Transit Classification Pipeline for Exoplanet Discovery
*   **Challenge Statement:** AI-based data analysis pipeline to automatically detect exoplanet transit signals from noisy astronomical light curves.
*   **Team Name:** [Enter Team Name]
*   **Members:** [Enter Member Names]
*   **Institution:** [Enter University/College Name]

---

### SLIDE 2: The Challenge
*   **The Scientific Gap:** Exoplanet detection via transit photometry requires identifying miniscule brightness dips in stars (down to 0.01%).
*   **Noise Contaminations:** 
    *   *Stellar Blending:* Contamination from foreground/background stars in crowded fields.
    *   *Detector Response:* High-frequency instrumental noise and random cosmic ray outliers.
    *   *False Positives:* Eclipsing binary stars (mimic transits but are stellar-sized) and active starspots (periodic rotational variations).
*   **Limitations of Traditional Methods:** Algorithms like Box Least Squares (BLS) fail in low Signal-to-Noise Ratio (SNR) regimes and are computationally slow.

---

### SLIDE 3: Proposed Solution - "AstroPulse" Pipeline
We propose a **hybrid neural pipeline** combining statistical signal detrending, unsupervised deep denoising, and supervised 1D convolutional classification.
*   **Phase 1: Pre-processing:** 3-sigma clipping for flares + running median filter for stellar detrending.
*   **Phase 2: Denoising Autoencoder (DAE):** 1D convolutional autoencoder that reconstructs the clean underlying transit profile by stripping away residual instrument noise.
*   **Phase 3: 1D CNN Classifier:** A deep classifier with Global Average Pooling that evaluates denoised light curves to output a transit probability score (0.0 to 1.0), trained to distinguish planets from binary stars and stellar rotation.
*   **Phase 4: Transit Fitting:** Phase-folding the light curve at the detected period and fitting a trapezoidal transit model to calculate physical properties (radius, period).

---

### SLIDE 4: Technology Stack & Pipeline Flow
*   **Languages & Frameworks:** Python, PyTorch (Machine Learning), React & Vite (Frontend Dashboard).
*   **Astronomy Tools:** Lightkurve API, Astropy (fits file ingestion and coordinate systems), Pandas, NumPy.
*   **Model Architectures:**
    *   *DAE:* 3-layer 1D Conv Encoder + 3-layer 1D ConvTranspose Decoder.
    *   *Classifier:* 4-layer Conv1D feature extractor + Batch Normalization + Linear classification head.
*   **Deployment:** Dockerized microservices, model inference accelerated via NVIDIA TensorRT.

---

### SLIDE 5: AI Model Training Details
*   **Dataset:** Pre-processed Kepler and TESS light curves.
*   **Data Augmentation:** Synthetic transit injections into real active star datasets to address class imbalance (since exoplanets are extremely rare, <1% of stars).
*   **Training Specs (10,000 samples):**
    *   *GPU (e.g. NVIDIA T4):* ~30 minutes of training.
    *   *Optimizer:* Adam (learning_rate=0.001).
    *   *Evaluation Metrics:* Validation Accuracy (Target: >96%), F1-Score (Target: >0.95) to minimize False Positives.

---

### SLIDE 6: Feasibility, Scale & Impact
*   **Computational Feasibility:** Pre-processing detrends data in O(N). Trained neural networks evaluate light curves in milliseconds, compared to traditional grid searches which take minutes per star.
*   **Scalability:** Perfect for next-gen massive space telescopes (e.g., PLATO, LSST, and ISRO's future space science missions) generating petabytes of telemetry.
*   **Impact on Space Science:** Automates the vetting process, reducing the time for exoplanet validation from months to seconds, and helping Indian space scientists discover Earth-like habitable planets.`;

  const copyToClipboard = () => {
    navigator.clipboard.writeText(proposalText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadFile = () => {
    const element = document.createElement("a");
    const file = new Blob([proposalText], { type: "text/markdown" });
    element.href = URL.createObjectURL(file);
    element.download = "isro_hackathon_proposal.md";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="card glass-card border border-slate-800 rounded-2xl p-6 max-w-4xl mx-auto space-y-6">
      {/* Header controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-800 gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-blue-950/80 border border-blue-500/30 rounded-xl text-blue-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">Idea Submission Proposal</h3>
            <p className="text-xs text-slate-400">Review, customize, and export your slides for the hackathon portal</p>
          </div>
        </div>

        <div className="flex items-center space-x-3 sm:self-center">
          <button 
            onClick={copyToClipboard}
            className="flex items-center space-x-1.5 px-4 py-2 bg-slate-900 border border-slate-700 hover:border-slate-500 rounded-xl text-xs font-semibold text-slate-200 transition-all cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-400" />
                <span>Copy Slide Content</span>
              </>
            )}
          </button>

          <button 
            onClick={downloadFile}
            className="flex items-center space-x-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-xl text-xs font-bold text-white transition-all cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download .md</span>
          </button>
        </div>
      </div>

      {/* Slide Text Preview */}
      <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-900 overflow-y-auto max-h-[500px] scrollbar-thin scrollbar-thumb-slate-800">
        <pre className="font-sans text-xs text-slate-300 whitespace-pre-wrap leading-relaxed space-y-4">
          <div className="text-center font-bold text-base text-cyan-400 border border-cyan-500/20 py-3 rounded-lg bg-cyan-950/10 mb-6">
            BHARATIYA ANTARIKSH HACKATHON 2026 PROPOSAL
          </div>
          
          <div className="space-y-6">
            <div>
              <span className="text-cyan-400 font-bold block text-sm border-b border-slate-800 pb-1 mb-2">SLIDE 1: Title & Team Details</span>
              <p className="pl-4 font-mono text-[11px] text-slate-400">
                <strong>Project Title:</strong> AstroPulse: Deep Denoising & Transit Classification Pipeline for Exoplanet Discovery<br />
                <strong>Challenge Statement:</strong> AI-based data analysis pipeline to automatically detect exoplanet transit signals from noisy astronomical light curves.<br />
                <strong>Team Info:</strong> Customize with your Team Name, Members, and College in your PPT template.
              </p>
            </div>

            <div>
              <span className="text-cyan-400 font-bold block text-sm border-b border-slate-800 pb-1 mb-2">SLIDE 2: The Challenge (Stellar Noise & Blending)</span>
              <p className="pl-4 font-mono text-[11px] text-slate-400">
                • <strong>The Scientific Gap:</strong> Exoplanet detection via transit photometry requires identifying miniscule brightness dips in stars (down to 0.01%).<br />
                • <strong>Noise Contaminations:</strong> Stellar blending from foreground/background stars in crowded fields, high-frequency instrument noise, and cosmic ray outliers.<br />
                • <strong>False Positives:</strong> Eclipsing binary stars (V-shape) and active rotating starspots (sinusoidal) easily confuse classical detectors.
              </p>
            </div>

            <div>
              <span className="text-cyan-400 font-bold block text-sm border-b border-slate-800 pb-1 mb-2">SLIDE 3: Proposed Architecture (Hybrid AI Pipeline)</span>
              <p className="pl-4 font-mono text-[11px] text-slate-400">
                • <strong>1. Digital Preprocessing:</strong> Outlier removal + running median filter detrending.<br />
                • <strong>2. Deep Denoising (DAE):</strong> Conv1D Autoencoder reconstructs the clean underlying transit profile by stripping away residual instrument noise.<br />
                • <strong>3. 1D CNN Classifier:</strong> Deep classifier with Global Average Pooling evaluates denoised light curves to output a transit probability score (0.0 to 1.0).<br />
                • <strong>4. Folding & Parameter Fitting:</strong> Fold light curve at the detected period and fit a trapezoidal model to calculate physical exoplanet dimensions.
              </p>
            </div>

            <div>
              <span className="text-cyan-400 font-bold block text-sm border-b border-slate-800 pb-1 mb-2">SLIDE 4: Technology Stack</span>
              <p className="pl-4 font-mono text-[11px] text-slate-400">
                • <strong>Backend:</strong> Python, PyTorch (Deep Learning), AstroPy & Lightkurve (Astronomical ingestion).<br />
                • <strong>Frontend:</strong> React, TailwindCSS, Vite, Recharts (Telemetry UI).<br />
                • <strong>Architectures:</strong> Conv1D DAE (142,657 parameters) & Conv1D Classifier (89,321 parameters).
              </p>
            </div>

            <div>
              <span className="text-cyan-400 font-bold block text-sm border-b border-slate-800 pb-1 mb-2">SLIDE 5: AI Training & Optimization</span>
              <p className="pl-4 font-mono text-[11px] text-slate-400">
                • <strong>Imbalance Fix:</strong> Synthetic transit injections into real active star datasets to resolve exoplanet rarity.<br />
                • <strong>GPU Training:</strong> Takes &lt;30 mins for 10,000 samples on a single T4 GPU.<br />
                • <strong>Evaluation Targets:</strong> Validation Accuracy &gt;96%, F1-Score &gt;0.95 to minimize false positive triggers.
              </p>
            </div>

            <div>
              <span className="text-cyan-400 font-bold block text-sm border-b border-slate-800 pb-1 mb-2">SLIDE 6: Impact & Scalability</span>
              <p className="pl-4 font-mono text-[11px] text-slate-400">
                • <strong>Computational Feasibility:</strong> Preprocessing and CNN inference run in milliseconds, compared to traditional grid searches which take minutes per star.<br />
                • <strong>Scale:</strong> Pre-trained weights enable real-time telemetry processing for future space missions (PLATO, LSST, etc.).<br />
                • <strong>Scientific Yield:</strong> Drastically speeds up exoplanet vetting, allowing Indian scientists to discover habitable Earth analogs faster.
              </p>
            </div>
          </div>
        </pre>
      </div>

      <div className="bg-blue-950/20 border border-blue-500/20 rounded-xl p-4 text-xs text-slate-300">
        <strong>💡 Hackathon Tip:</strong> Copy this content and paste it directly into your PowerPoint slides. Keep the layout visual—use flowcharts for the pipeline and display screenshots of this running React prototype to impress the reviewers.
      </div>
    </div>
  );
}
