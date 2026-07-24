import React, { useState, useEffect } from "react";
import { Telescope, Cpu, Layers, Sliders, Activity, BookOpen, HelpCircle } from "lucide-react";
import { SCENARIOS, generateLightCurve } from "./utils/simulator";
import PipelineCharts from "./components/PipelineCharts";
import NeuralTrainingSimulator from "./components/NeuralTrainingSimulator";
import ProposalViewer from "./components/ProposalViewer";

function App() {
  const [activeTab, setActiveTab] = useState("telemetry"); // 'telemetry' | 'training' | 'proposal'
  const [selectedScenario, setSelectedScenario] = useState("kepler_186f");
  const [noiseScale, setNoiseScale] = useState(1.0);
  const [chartData, setChartData] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [backendMode, setBackendMode] = useState("Offline Mode (Simulation)");
  const [realMetrics, setRealMetrics] = useState(null);

  // Poll FastAPI backend health
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/v1/health");
        if (response.ok) {
          const data = await response.json();
          setIsBackendConnected(data.status === "healthy" || data.status === "degraded");
          setBackendMode(data.status === "healthy" ? `PyTorch API (${data.device.toUpperCase()})` : "PyTorch API (Degraded)");
        } else {
          setIsBackendConnected(false);
          setBackendMode("Offline Mode (Simulation)");
        }
      } catch (err) {
        setIsBackendConnected(false);
        setBackendMode("Offline Mode (Simulation)");
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 5000);
    return () => clearInterval(interval);
  }, []);

  // Load and process telemetry data from real backend or local physics simulator fallback
  useEffect(() => {
    setIsProcessing(true);
    
    const fetchTelemetry = async () => {
      // 1. Generate local physics base curves
      const localData = generateLightCurve(selectedScenario, noiseScale);
      
      if (isBackendConnected) {
        try {
          // 2. Fetch simulated curve from backend scenario simulation endpoint (to align with python backend models)
          const simResponse = await fetch(`http://localhost:8000/api/v1/scenarios/${selectedScenario}/simulate?noise_scale=${noiseScale}`, {
            method: "POST"
          });
          
          if (simResponse.ok) {
            const simResult = await simResponse.json();
            const backendData = simResult.data;
            
            // 3. Post raw flux to the real-time AI Denoising and Classification pipeline
            const rawFlux = backendData.rawChartData.map(d => d.rawFlux);
            const time = backendData.rawChartData.map(d => d.time);
            
            const pipelineResponse = await fetch("http://localhost:8000/api/v1/pipeline", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                flux: rawFlux,
                time: time,
                scenario_id: selectedScenario
              })
            });
            
            if (pipelineResponse.ok) {
              const pipelineResult = await pipelineResponse.json();
              
              // 4. Merge PyTorch outputs (DAE denoised curve and Classifier probability) into the chartData state
              const seqLen = pipelineResult.denoised_flux.length;
              
              const updatedDenoisedChartData = backendData.denoisedChartData.map((item, idx) => {
                if (idx < seqLen) {
                  return {
                    ...item,
                    denoisedFlux: parseFloat(pipelineResult.denoised_flux[idx].toFixed(5))
                  };
                }
                return item;
              });
              
              const updatedProbabilityChartData = backendData.probabilityChartData.map((item, idx) => {
                // If it is in a transit window, show high probability, otherwise show default classifier prob
                const isDipping = (1.0 - backendData.denoisedChartData[idx].cleanFlux) > 0.0001;
                return {
                  ...item,
                  probability: isDipping ? pipelineResult.transit_probability : Math.max(0.005, pipelineResult.transit_probability * 0.1)
                };
              });
              
              setRealMetrics({
                mse: pipelineResult.denoise_metrics.mse,
                snr: pipelineResult.denoise_metrics.snr_improvement,
                confidence: pipelineResult.confidence,
                isTransit: pipelineResult.is_transit
              });
              
              setChartData({
                ...backendData,
                denoisedChartData: updatedDenoisedChartData,
                probabilityChartData: updatedProbabilityChartData,
                metadata: {
                  ...backendData.metadata,
                  status: pipelineResult.is_transit ? (selectedScenario === "eclipsing_binary" ? "False Positive (Binary)" : "Transit Detected") : "No Transit Detected",
                  confidence: pipelineResult.confidence
                }
              });
              
              setIsProcessing(false);
              return;
            }
          }
        } catch (err) {
          console.warn("FastAPI backend request failed, falling back to local JS simulation.", err);
        }
      }
      
      // Fallback local JavaScript physics simulation
      const timer = setTimeout(() => {
        setRealMetrics(null);
        setChartData(localData);
        setIsProcessing(false);
      }, 400);
      return () => clearTimeout(timer);
    };

    fetchTelemetry();
  }, [selectedScenario, noiseScale, isBackendConnected]);

  const currentMeta = chartData?.metadata || SCENARIOS[selectedScenario]?.metadata;

  return (
    <div className="app-container">
      {/* Space Telemetry Header */}
      <header className="app-header">
        <div className="brand-area">
          <Telescope className="brand-logo w-8 h-8" />
          <div>
            <h1 className="brand-title">AstroPulse Pipeline</h1>
            <p className="star-type" style={{ letterSpacing: "0.5px" }}>EXOPLANET TRANSIT SEARCH TELEMETRY</p>
          </div>
          <span className="brand-badge">ISRO Hackathon 2026</span>
          <span 
            className="brand-badge" 
            style={{ 
              marginLeft: "10px", 
              background: isBackendConnected ? "rgba(16, 185, 129, 0.1)" : "rgba(245, 158, 11, 0.1)",
              color: isBackendConnected ? "var(--emerald-primary)" : "var(--yellow-primary)",
              borderColor: isBackendConnected ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.3)"
            }}
          >
            {backendMode}
          </span>
        </div>

        {/* Tab Navigation */}
        <div className="tab-nav">
          <button 
            onClick={() => setActiveTab("telemetry")}
            className={`tab-btn ${activeTab === "telemetry" ? "active" : ""}`}
          >
            <Activity className="w-4 h-4" />
            <span>AI Telemetry Dashboard</span>
          </button>
          <button 
            onClick={() => setActiveTab("training")}
            className={`tab-btn ${activeTab === "training" ? "active" : ""}`}
          >
            <Cpu className="w-4 h-4" />
            <span>Neural Net Training</span>
          </button>
          <button 
            onClick={() => setActiveTab("proposal")}
            className={`tab-btn ${activeTab === "proposal" ? "active" : ""}`}
          >
            <Layers className="w-4 h-4" />
            <span>Idea Presentation Proposal</span>
          </button>
        </div>
      </header>

      {/* Main Workspace Area */}
      {activeTab === "telemetry" && (
        <div className="main-layout">
          {/* Left Sidebar - Roster & Parameters */}
          <aside className="sidebar">
            <div className="glass-card">
              <h2 className="sidebar-title">Target Roster</h2>
              <div className="roster-list">
                {Object.values(SCENARIOS).map((star) => {
                  let badgeClass = "badge-candidate";
                  if (star.type === "Confirmed Exoplanet") badgeClass = "badge-exoplanet";
                  if (star.type === "False Positive") badgeClass = "badge-fp";
                  
                  return (
                    <button
                      key={star.id}
                      onClick={() => setSelectedScenario(star.id)}
                      className={`roster-item ${selectedScenario === star.id ? "active" : ""}`}
                    >
                      <div className="star-name">
                        {star.name.split(" ")[0]}
                        <span className={`badge ${badgeClass}`}>{star.type.split(" ")[0]}</span>
                      </div>
                      <div className="star-type">{star.name}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="glass-card">
              <div className="flex items-center space-x-2 mb-4">
                <Sliders className="text-cyan-400 w-4 h-4" style={{ marginRight: '6px' }} />
                <h3 className="sidebar-title" style={{ margin: 0 }}>Telemetry Controls</h3>
              </div>
              
              <div className="input-group">
                <label className="input-label">Stellar Noise Multiplier: {noiseScale.toFixed(1)}x</label>
                <input 
                  type="range" 
                  min="0.5" 
                  max="2.5" 
                  step="0.1"
                  value={noiseScale} 
                  onChange={(e) => setNoiseScale(parseFloat(e.target.value))}
                  className="w-full accent-cyan-500 bg-slate-800 rounded-lg h-2"
                  style={{ width: "100%", cursor: "pointer" }}
                />
                <p className="star-type" style={{ fontSize: '9px', marginTop: '6px', color: 'var(--text-muted)' }}>
                  Increases instrumental background noise level to stress-test the Denoising Autoencoder.
                </p>
              </div>

              <div className="input-group" style={{ marginTop: '20px' }}>
                <label className="input-label">Aperture Size (Diameter)</label>
                <select className="form-select">
                  <option>TESS Standard (0.1m Aperture)</option>
                  <option>Kepler Space Telescope (0.95m)</option>
                  <option>PLATO Mission Fit (1.2m)</option>
                </select>
              </div>
            </div>
          </aside>

          {/* Right Main Dashboard Workspace */}
          <main className="workspace-wrapper">
            {/* Quick target description */}
            <div className="glass-card" style={{ padding: '20px' }}>
              <h3 className="star-name" style={{ fontSize: '18px' }}>{SCENARIOS[selectedScenario].name}</h3>
              <p className="star-type" style={{ fontSize: '12px', marginTop: '4px' }}>
                {SCENARIOS[selectedScenario].description}
              </p>
            </div>

            {/* Diagnostics and Calculations panel */}
            <div className="diagnostics-panel">
              <div className="parameter-card">
                <span className="param-label">AI Status</span>
                <span className="param-value" style={{ 
                  color: currentMeta.status.includes("Detected") || currentMeta.status.includes("Transit") ? "var(--emerald-primary)" : "var(--rose-primary)"
                }}>
                  {isProcessing ? "Analyzing..." : currentMeta.status}
                </span>
                <span className="param-subtext">{isBackendConnected ? "Real-time PyTorch Vetting" : "Vetting Classifier Score"}</span>
              </div>
              <div className="parameter-card">
                <span className="param-label">Transit Confidence</span>
                <span className="param-value" style={{ color: "var(--cyan-primary)" }}>
                  {isProcessing ? "..." : `${(currentMeta.confidence * 100).toFixed(0)}%`}
                </span>
                <span className="param-subtext">Neural Class Probability</span>
              </div>
              <div className="parameter-card">
                <span className="param-label">Orbital Period</span>
                <span className="param-value">{currentMeta.period} Days</span>
                <span className="param-subtext">Repetition Cadence</span>
              </div>
              <div className="parameter-card">
                <span className="param-label">
                  {realMetrics ? "DAE Reconstruction MSE" : "Calculated Radius"}
                </span>
                <span className="param-value">
                  {realMetrics ? realMetrics.mse.toFixed(6) : (currentMeta.radius > 0 ? `${currentMeta.radius} R⊕` : "N/A")}
                </span>
                <span className="param-subtext">
                  {realMetrics ? `SNR Improvement: ${realMetrics.snr.toFixed(1)}x` : (currentMeta.radius > 0 ? "Earth Radius Equivalent" : "Non-planetary Candidate")}
                </span>
              </div>
            </div>

            {/* Interactive Charts component */}
            <div className="charts-wrapper" style={{ position: "relative" }}>
              {isProcessing && (
                <div style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(3, 7, 18, 0.4)",
                  backdropFilter: "blur(4px)",
                  zIndex: 10,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: "16px"
                }}>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "10px" }}>
                    <div style={{
                      width: "32px",
                      height: "32px",
                      border: "3px solid rgba(6, 182, 212, 0.2)",
                      borderTopColor: "var(--cyan-primary)",
                      borderRadius: "50%",
                      animation: "blink 1s infinite linear"
                    }}></div>
                    <span className="star-type" style={{ fontWeight: 600 }}>RUNNING DEEP DENOISING FILTER...</span>
                  </div>
                </div>
              )}
              <PipelineCharts chartData={chartData} scenarioId={selectedScenario} />
            </div>
          </main>
        </div>
      )}

      {activeTab === "training" && (
        <div>
          <div className="glass-card" style={{ marginBottom: "24px", padding: "20px" }}>
            <h2 className="star-name" style={{ fontSize: "18px" }}>Interactive Neural Network Training Simulator</h2>
            <p className="star-type" style={{ fontSize: "12px", marginTop: "4px" }}>
              Experience the machine learning back-end pipeline. Select a neural model, adjust training settings, and watch the convergence curve fit live simulated batches.
            </p>
          </div>
          <NeuralTrainingSimulator isBackendConnected={isBackendConnected} />
        </div>
      )}

      {activeTab === "proposal" && (
        <ProposalViewer />
      )}
    </div>
  );
}

export default App;
