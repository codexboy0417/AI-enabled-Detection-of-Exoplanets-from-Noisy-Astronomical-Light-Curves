import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from "recharts";
import { Activity, ShieldAlert, Cpu, ZoomIn } from "lucide-react";

export default function PipelineCharts({ chartData, scenarioId }) {
  if (!chartData) {
    return (
      <div className="flex h-96 items-center justify-center border border-dashed border-slate-800 rounded-2xl text-slate-500">
        <span>Select a star from the telemetry roster to initialize pipeline stream</span>
      </div>
    );
  }

  const { rawChartData, denoisedChartData, probabilityChartData, phaseFoldedData } = chartData;

  // Custom tooltips to look sleek and fit the deep space theme
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900 border border-slate-700 px-3 py-2 rounded-lg text-xs font-mono shadow-xl">
          <p className="text-slate-400 font-bold mb-1">Time: {label} days</p>
          {payload.map((item, index) => (
            <p key={index} style={{ color: item.stroke || item.fill }}>
              {item.name}: {item.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const CustomFoldedTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900 border border-slate-700 px-3 py-2 rounded-lg text-xs font-mono shadow-xl">
          <p className="text-slate-400 font-bold mb-1">Phase: {label}</p>
          {payload.map((item, index) => (
            <p key={index} style={{ color: item.color || item.stroke }}>
              {item.name}: {item.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* 2x2 Grid of the Pipeline Stages */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        
        {/* Stage 1: Noisy Light Curve */}
        <div className="card glass-card p-5 border border-slate-800 rounded-2xl h-[320px] flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="text-rose-400 w-4 h-4" />
              <h4 className="text-xs font-bold tracking-wider text-slate-300 uppercase">1. Raw Telemetry Light Curve</h4>
            </div>
            <p className="text-[11px] text-slate-400">
              Raw flux measurements over time, containing detector noise, positive cosmic ray outliers, and stellar trend.
            </p>
          </div>
          
          <div className="w-full h-[210px] mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rawChartData} margin={{ left: -15, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" tickMargin={5} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                <YAxis stroke="#64748b" domain={["auto", "auto"]} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: "10px", fontFamily: "monospace", paddingTop: "5px" }} />
                <Line name="Raw Flux" type="monotone" dataKey="rawFlux" stroke="#f43f5e" strokeWidth={1} dot={false} />
                <Line name="Stellar Trend" type="monotone" dataKey="trend" stroke="#eab308" strokeWidth={1.5} dot={false} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Stage 2: AI Denoised Light Curve */}
        <div className="card glass-card p-5 border border-slate-800 rounded-2xl h-[320px] flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <Cpu className="text-cyan-400 w-4 h-4" />
              <h4 className="text-xs font-bold tracking-wider text-slate-300 uppercase">2. AI Denoised Light Curve (DAE)</h4>
            </div>
            <p className="text-[11px] text-slate-400">
              Neural autoencoder reconstruction: filters high-frequency noise and removes trend to isolate transit signals.
            </p>
          </div>
          
          <div className="w-full h-[210px] mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={denoisedChartData} margin={{ left: -15, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" tickMargin={5} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                <YAxis stroke="#64748b" domain={["auto", "auto"]} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: "10px", fontFamily: "monospace", paddingTop: "5px" }} />
                <Line name="Detrended Raw" type="monotone" dataKey="rawFlux" stroke="#475569" strokeWidth={0.8} dot={false} />
                <Line name="Denoised (DAE)" type="monotone" dataKey="denoisedFlux" stroke="#06b6d4" strokeWidth={1.5} dot={false} />
                <Line name="Physics Model" type="monotone" dataKey="cleanFlux" stroke="#a855f7" strokeWidth={1} dot={false} strokeDasharray="4 4" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Stage 3: Transit Probability Over Time */}
        <div className="card glass-card p-5 border border-slate-800 rounded-2xl h-[320px] flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <ShieldAlert className="text-emerald-400 w-4 h-4" />
              <h4 className="text-xs font-bold tracking-wider text-slate-300 uppercase">3. Rolling Transit Probability</h4>
            </div>
            <p className="text-[11px] text-slate-400">
              Output probability of 1D CNN classifier run in a sliding window over the denoised curve.
            </p>
          </div>
          
          <div className="w-full h-[210px] mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={probabilityChartData} margin={{ left: -15, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" tickMargin={5} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                <YAxis stroke="#64748b" domain={[0, 1.0]} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: "10px", fontFamily: "monospace", paddingTop: "5px" }} />
                <Line name="Detection Confidence" type="monotone" dataKey="probability" stroke="#10b981" strokeWidth={2} dot={false} />
                <ReferenceLine y={0.5} label={{ value: "Detection Threshold", fill: "#ef4444", fontSize: 9, position: "top", fontFamily: "monospace" }} stroke="#ef4444" strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Stage 4: Phase-Folded Light Curve Fit */}
        <div className="card glass-card p-5 border border-slate-800 rounded-2xl h-[320px] flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <ZoomIn className="text-yellow-400 w-4 h-4" />
              <h4 className="text-xs font-bold tracking-wider text-slate-300 uppercase">4. Phase-Folded Data & Fitting</h4>
            </div>
            <p className="text-[11px] text-slate-400">
              Data folded modulo the detected orbital period, overlaying the best-fit exoplanet transit profile.
            </p>
          </div>
          
          <div className="w-full h-[210px] mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={phaseFoldedData} margin={{ left: -15, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="phase" stroke="#64748b" tickMargin={5} tickFormatter={(v) => v.toFixed(2)} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                <YAxis stroke="#64748b" domain={["auto", "auto"]} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                <Tooltip content={<CustomFoldedTooltip />} />
                <Legend wrapperStyle={{ fontSize: "10px", fontFamily: "monospace", paddingTop: "5px" }} />
                {/* Scatter dot plot simulation in Recharts */}
                <Line name="Folded Data" type="monotone" dataKey="rawFlux" stroke="none" fill="#38bdf8" dot={{ r: 1.5, fill: "#38bdf8", strokeWidth: 0 }} />
                <Line name="Model Fit" type="monotone" dataKey="modelFlux" stroke="#eab308" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}
