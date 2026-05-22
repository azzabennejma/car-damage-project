import { useState, useRef, useCallback } from "react";

// ── Config ──────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";;

const CLASS_COLORS = {
  scratch:          { bg: "#ef4444", icon: "✕✕" },
  dent:             { bg: "#f59e0b", icon: "◎"  },
  crack:            { bg: "#8b5cf6", icon: "╲╱" },
  "glass shatter": { bg: "#3b82f6", icon: "◇"  },
  "lamp broken":  { bg: "#ec4899", icon: "○"  },
  "tire flat":      { bg: "#10b981", icon: "◉"  },
};

const confColor = (c) =>
  c >= 0.85 ? "#ef4444" : c >= 0.70 ? "#f59e0b" : "#10b981";

const fmtTime = (date) =>
  date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

// ── DonutChart ───────────────────────────────────────────────
function DonutChart({ value }) {
  const r = 38, cx = 50, cy = 50;
  const circ = 2 * Math.PI * r;
  const segments = [
    { pct: value / 100,               color: "#1d6ae5" },
    { pct: (100 - value) * 0.4 / 100, color: "#ef4444" },
    { pct: (100 - value) * 0.3 / 100, color: "#f59e0b" },
    { pct: (100 - value) * 0.3 / 100, color: "#10b981" },
  ];
  let cumulative = 0;
  return (
    <svg width={100} height={100} viewBox="0 0 100 100">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#21262d" strokeWidth={9} />
      {segments.map((seg, i) => {
        const dash = circ * seg.pct;
        const gap  = circ - dash;
        const el   = (
          <circle key={i} cx={cx} cy={cy} r={r} fill="none"
            stroke={seg.color} strokeWidth={9}
            strokeDasharray={`${dash} ${gap}`}
            strokeDashoffset={-(cumulative * circ)}
            transform={`rotate(-90 ${cx} ${cy})`}
            strokeLinecap="butt"
          />
        );
        cumulative += seg.pct;
        return el;
      })}
      <text x={cx} y={cy - 5} textAnchor="middle"
        fill="#8b949e" fontSize={9} fontFamily="sans-serif">mAP</text>
      <text x={cx} y={cy + 11} textAnchor="middle"
        fill="#e6edf3" fontSize={15} fontWeight={700} fontFamily="sans-serif">
        {value}%
      </text>
    </svg>
  );
}

// ── HistoryItem ──────────────────────────────────────────────
function HistoryItem({ item, onSelect, isSelected }) {
  const topDet = item.detections[0];
  const col = topDet ? (CLASS_COLORS[topDet.class]?.bg || "#ef4444") : "#484f58";
  return (
    <div
      onClick={() => onSelect(item)}
      style={{
        display: "flex", alignItems: "center", gap: 10,
        padding: "10px 12px", borderRadius: 8, cursor: "pointer",
        background: isSelected ? "#1d6ae522" : "transparent",
        border: isSelected ? "1px solid #1d6ae555" : "1px solid transparent",
        transition: "all .15s", marginBottom: 6,
      }}
    >
      <img
        src={`data:image/jpeg;base64,${item.image_base64}`}
        alt="history"
        style={{ width: 48, height: 40, objectFit: "cover", borderRadius: 5, border: "1px solid #21262d", flexShrink: 0 }}
      />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3", fontFamily: "sans-serif",
          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {item.filename}
        </div>
        <div style={{ fontSize: 10, color: "#8b949e", fontFamily: "sans-serif", marginTop: 2 }}>
          {item.detections.length} detection{item.detections.length !== 1 ? "s" : ""} · {item.infer_ms}ms
        </div>
        <div style={{ fontSize: 10, color: "#484f58", fontFamily: "sans-serif" }}>
          {fmtTime(new Date(item.timestamp))}
        </div>
      </div>
      {topDet && (
        <div style={{
          padding: "2px 7px", borderRadius: 5, fontSize: 10,
          fontWeight: 700, fontFamily: "monospace",
          background: col, color: "#fff", flexShrink: 0,
        }}>
          {topDet.confidence.toFixed(2)}
        </div>
      )}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────
export default function CarDamageDetection() {
  const [file, setFile]             = useState(null);
  const [preview, setPreview]       = useState(null);
  const [predicted, setPredicted]   = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading]       = useState(false);
  const [inferTime, setInferTime]   = useState(null);
  const [conf, setConf]             = useState(0.25);
  const [dragging, setDragging]     = useState(false);
  const [tab, setTab]               = useState("dashboard");
  const [error, setError]           = useState(null);
  const [serverOk, setServerOk]     = useState(null);
  const [history, setHistory]       = useState([]);       // ← stores all past inferences
  const [selectedHistory, setSelectedHistory] = useState(null);

  const inputRef = useRef();

  // ── File handling ─────────────────────────────────────────
  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setPredicted(null);
    setDetections([]);
    setInferTime(null);
    setError(null);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  // ── Server health check ───────────────────────────────────
  const checkServer = async () => {
    try {
      const res  = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setServerOk(data.status === "ok");
    } catch {
      setServerOk(false);
    }
  };

  // ── Real inference ────────────────────────────────────────
  const runInference = async () => {
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    setPredicted(null);
    setDetections([]);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_BASE}/predict?conf=${conf}`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(`Server error ${res.status}: ${msg}`);
      }

      // Parse JSON response — contains base64 image + detections
      const data = await res.json();

      const imgSrc = `data:image/jpeg;base64,${data.image_base64}`;
      setPredicted(imgSrc);
      setDetections(data.detections);
      setInferTime(data.infer_ms);

      // ── Store in history ──────────────────────────────────
      const historyEntry = {
        id:           Date.now(),
        timestamp:    Date.now(),
        filename:     file.name,
        detections:   data.detections,
        image_base64: data.image_base64,
        infer_ms:     data.infer_ms,
        conf:         data.conf,
      };
      setHistory((prev) => [historyEntry, ...prev]);

    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? `Cannot reach FastAPI at ${API_BASE}. Make sure the server is running:\n  uvicorn main:app --reload --port 8000`
          : err.message
      );
    }

    setLoading(false);
  };

  // ── Load a history item into the result panel ─────────────
  const loadFromHistory = (item) => {
    setSelectedHistory(item.id);
    setPredicted(`data:image/jpeg;base64,${item.image_base64}`);
    setDetections(item.detections);
    setInferTime(item.infer_ms);
    setError(null);
  };

  // ── Shared styles ─────────────────────────────────────────
  const card = {
    background: "#161b22", border: "1px solid #21262d",
    borderRadius: 10, padding: "1rem",
  };
  const secTitle = {
    fontSize: 13, fontWeight: 700, color: "#e6edf3",
    marginBottom: ".875rem", display: "flex", alignItems: "center",
    gap: 7, fontFamily: "sans-serif",
  };
  const titleIcon = {
    width: 26, height: 26, background: "#1d6ae5", borderRadius: 6,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 12, flexShrink: 0,
  };
  const chip = (col, bg) => ({
    padding: "4px 12px", borderRadius: 16, fontSize: 11,
    fontWeight: 600, fontFamily: "sans-serif", background: bg, color: col,
  });

  // ── Render ────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: "100vh", background: "#0d1117", color: "#e6edf3",
      fontFamily: "'DM Mono','Courier New',monospace",
      display: "flex", flexDirection: "column",
    }}>

      {/* ── Navbar ─────────────────────────────────────────── */}
      <nav style={{
        background: "#161b22", borderBottom: "1px solid #21262d",
        padding: "0 1.5rem", display: "flex", alignItems: "center",
        gap: "1rem", height: 56, flexShrink: 0,
      }}>
        <div style={{ width: 34, height: 34, background: "#1d6ae5", borderRadius: 8,
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0 }}>
          🚗
        </div>
        <div style={{ marginRight: "auto" }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: "#e6edf3", whiteSpace: "nowrap" }}>
            Car Damage <span style={{ color: "#1d6ae5" }}>Detection</span>
          </div>
          <div style={{ fontSize: 10, color: "#8b949e", fontFamily: "sans-serif" }}>
            AI-Powered using YOLOv8m + FastAPI
          </div>
        </div>

        {[
          { id: "dashboard", label: "Dashboard" },
          { id: "history",   label: `History${history.length > 0 ? ` (${history.length})` : ""}` },
          { id: "model",     label: "Model" },
        ].map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: "5px 14px", borderRadius: 6, border: "none", cursor: "pointer",
            fontSize: 12, fontFamily: "sans-serif",
            background: tab === t.id ? "#1d6ae5" : "transparent",
            color:      tab === t.id ? "#fff"    : "#8b949e",
            fontWeight: tab === t.id ? 600       : 400,
          }}>
            {t.label}
          </button>
        ))}

        {/* Server status */}
        <div onClick={checkServer} title="Click to ping server"
          style={{ display: "flex", alignItems: "center", gap: 5, cursor: "pointer", marginLeft: 4 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: serverOk === null ? "#8b949e" : serverOk ? "#3fb950" : "#ef4444",
          }} />
          <span style={{ fontSize: 11, color: "#8b949e", fontFamily: "sans-serif" }}>
            {serverOk === null ? "Check server" : serverOk ? "API online" : "API offline"}
          </span>
        </div>
      </nav>

      {/* ── Dashboard tab ──────────────────────────────────── */}
      {tab === "dashboard" && (
        <div style={{
          display: "grid", gridTemplateColumns: "260px 1fr 260px",
          gap: "1rem", padding: "1rem", flex: 1, alignItems: "start",
        }}>

          {/* LEFT — Upload */}
          <div style={card}>
            <div style={secTitle}>
              <div style={titleIcon}>⬆</div>
              Upload Image
            </div>

            {/* Dropzone */}
            <div
              onClick={() => inputRef.current.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              style={{
                border: `2px dashed ${dragging ? "#1d6ae5" : "#30363d"}`,
                borderRadius: 8, padding: "1.5rem 1rem", textAlign: "center",
                cursor: "pointer", marginBottom: ".75rem",
                background: dragging ? "rgba(29,106,229,.06)" : "transparent",
                transition: "all .2s",
              }}
            >
              <div style={{ fontSize: 28, color: "#1d6ae5", marginBottom: 6 }}>☁</div>
              <div style={{ fontSize: 12, color: "#8b949e", fontFamily: "sans-serif" }}>
                Drag &amp; drop your image here
              </div>
              <div style={{ fontSize: 11, color: "#484f58", fontFamily: "sans-serif" }}>or click to browse</div>
              <button
                onClick={(e) => { e.stopPropagation(); inputRef.current.click(); }}
                style={{
                  background: "#1d6ae5", color: "#fff", border: "none", borderRadius: 7,
                  padding: "8px 16px", fontSize: 12, fontWeight: 600, cursor: "pointer",
                  display: "flex", alignItems: "center", gap: 5,
                  margin: ".75rem auto 0", fontFamily: "sans-serif",
                }}
              >
                ⬆ Choose Image
              </button>
              <div style={{ fontSize: 10, color: "#484f58", marginTop: 6, fontFamily: "sans-serif" }}>
                JPG, PNG up to 10MB
              </div>
            </div>

            <input ref={inputRef} type="file" accept="image/*"
              style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />

            {/* Confidence slider */}
            <div style={{ marginBottom: ".625rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between",
                fontSize: 11, color: "#8b949e", marginBottom: 3, fontFamily: "sans-serif" }}>
                <span>Confidence threshold</span>
                <span style={{ color: "#e6edf3", fontWeight: 600 }}>{conf.toFixed(2)}</span>
              </div>
              <input type="range" min="10" max="90" step="5"
                value={Math.round(conf * 100)}
                onChange={(e) => setConf(e.target.value / 100)}
                style={{ width: "100%", accentColor: "#1d6ae5" }} />
            </div>

            {/* Preview */}
            {preview && (
              <div style={{ marginBottom: ".625rem" }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: "#8b949e",
                  fontFamily: "sans-serif", textTransform: "uppercase",
                  letterSpacing: ".06em", marginBottom: 5 }}>Preview</div>
                <img src={preview} alt="preview" style={{
                  width: "100%", borderRadius: 7, maxHeight: 150,
                  objectFit: "cover", border: "1px solid #21262d",
                }} />
              </div>
            )}

            {/* Run button */}
            <button
              disabled={!file || loading}
              onClick={runInference}
              style={{
                width: "100%", padding: "10px 0", fontSize: 13, fontWeight: 600,
                border: "none", borderRadius: 7, marginTop: ".75rem",
                fontFamily: "sans-serif", transition: "background .2s",
                cursor: file && !loading ? "pointer" : "not-allowed",
                background: file && !loading ? "#1d6ae5" : "#21262d",
                color:      file && !loading ? "#fff"    : "#484f58",
              }}
            >
              {loading ? "⏳ Processing..." : "▶  Analyze Image"}
            </button>

            {/* Error */}
            {error && (
              <div style={{
                background: "#2d1215", border: "1px solid #6e2c2c", borderRadius: 8,
                padding: ".75rem 1rem", fontSize: 11, color: "#fca5a5",
                fontFamily: "sans-serif", marginTop: ".75rem", lineHeight: 1.6,
                whiteSpace: "pre-line",
              }}>
                <strong>Error:</strong> {error}
              </div>
            )}
          </div>

          {/* CENTER — Prediction result */}
          <div style={card}>
            <div style={secTitle}>
              <div style={titleIcon}>◈</div>
              Prediction Result
            </div>

            <div style={{
              position: "relative", borderRadius: 6, overflow: "hidden",
              background: "#0d1117", minHeight: 280,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              {loading ? (
                <div style={{ textAlign: "center", color: "#8b949e" }}>
                  <div style={{ fontSize: 32, marginBottom: 10 }}>⏳</div>
                  <div style={{ fontSize: 12, fontFamily: "sans-serif" }}>Running YOLOv8m inference...</div>
                </div>
              ) : predicted ? (
                <img src={predicted} alt="Annotated prediction"
                  style={{ width: "100%", display: "block", borderRadius: 8 }} />
              ) : (
                <div style={{ textAlign: "center", color: "#484f58" }}>
                  <div style={{ fontSize: 40, marginBottom: 10 }}>🖼</div>
                  <div style={{ fontSize: 12, fontFamily: "sans-serif" }}>
                    Upload an image and click Analyze
                  </div>
                </div>
              )}
            </div>

            {/* Status chips */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: ".75rem", flexWrap: "wrap" }}>
              <div style={chip("#e6edf3", "#21262d")}>
                {detections.length} damage{detections.length !== 1 ? "s" : ""} detected
              </div>
              <div style={chip("#e6edf3", "#21262d")}>YOLOv8m</div>
              {inferTime && (
                <div style={chip("#3fb950", "#122d1f")}>
                  ✓ Processed in {inferTime}ms
                </div>
              )}
            </div>
          </div>

          {/* RIGHT — Predictions + Metrics + Version */}
          <div style={{ display: "flex", flexDirection: "column", gap: ".875rem" }}>

            {/* Detections list */}
            <div style={card}>
              <div style={secTitle}>
                <div style={titleIcon}>◈</div>
                Predictions
              </div>
              {detections.length === 0 ? (
                <div style={{ fontSize: 11, color: "#484f58", fontFamily: "sans-serif",
                  textAlign: "center", padding: ".75rem 0" }}>
                  {loading ? "Running inference..." : "No detections yet"}
                </div>
              ) : (
                detections.map((d, i) => {
                  const col  = CLASS_COLORS[d.class]?.bg   || "#ef4444";
                  const icon = CLASS_COLORS[d.class]?.icon || "◆";
                  return (
                    <div key={i} style={{
                      display: "flex", alignItems: "center", gap: 8, padding: "8px 0",
                      borderBottom: i < detections.length - 1 ? "1px solid #21262d" : "none",
                    }}>
                      <div style={{
                        width: 34, height: 34, background: col + "22",
                        border: `1px solid ${col}44`, borderRadius: 7,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 14, color: col, flexShrink: 0,
                      }}>
                        {icon}
                      </div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, fontFamily: "sans-serif",
                          color: "#e6edf3", textTransform: "capitalize" }}>
                          {d.class.charAt(0).toUpperCase() + d.class.slice(1)}
                        </div>
                        <div style={{ fontSize: 10, color: "#8b949e", fontFamily: "sans-serif" }}>
                          Confidence: {(d.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div style={{
                        marginLeft: "auto", padding: "2px 8px", borderRadius: 5,
                        fontSize: 11, fontWeight: 700, fontFamily: "monospace",
                        background: confColor(d.confidence), color: "#fff", flexShrink: 0,
                      }}>
                        {d.confidence.toFixed(2)}
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Model Performance */}
            <div style={card}>
              <div style={secTitle}>
                <div style={titleIcon}>◉</div>
                Model Performance
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: ".875rem", marginTop: ".5rem" }}>
                <DonutChart value={85.6} />
                <div style={{ flex: 1 }}>
                  {[
                    ["Precision", "88.2%", "#1d6ae5"],
                    ["Recall",    "83.1%", "#ef4444"],
                    ["F1-Score",  "85.6%", "#1d6ae5"],
                  ].map(([label, val, col]) => (
                    <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 5 }}>
                      <div style={{ width: 7, height: 7, borderRadius: "50%", background: col, flexShrink: 0 }} />
                      <span style={{ fontSize: 11, color: "#8b949e", fontFamily: "sans-serif", flex: 1 }}>{label}:</span>
                      <span style={{ fontSize: 11, fontWeight: 600, fontFamily: "sans-serif", color: "#e6edf3" }}>{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Model Version */}
            <div style={card}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: ".75rem" }}>
                <div style={{ ...secTitle, marginBottom: 0 }}>
                  <div style={titleIcon}>◈</div>
                  Model Version
                </div>
                <span style={{ background: "#1d6ae5", color: "#fff", fontSize: 10,
                  padding: "2px 8px", borderRadius: 5, fontWeight: 700, fontFamily: "monospace" }}>
                  v2.1
                </span>
              </div>
              <div style={{ background: "#0d1117", border: "1px solid #21262d", borderRadius: 7, padding: ".625rem .875rem" }}>
                {[
                  ["Dataset",  "CarDD v2"],
                  ["Backend",  "FastAPI"],
                  ["Conf",     conf.toFixed(2)],
                ].map(([k, v], i, arr) => (
                  <div key={k} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    fontSize: 11, padding: "4px 0", fontFamily: "sans-serif", color: "#8b949e",
                    borderBottom: i < arr.length - 1 ? "1px solid #21262d" : "none",
                  }}>
                    <span>{k}:</span>
                    <span style={{ color: "#e6edf3", fontWeight: 600 }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── History tab ─────────────────────────────────────── */}
      {tab === "history" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          {history.length === 0 ? (
            <div style={{ textAlign: "center", color: "#484f58", paddingTop: "4rem" }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>🕓</div>
              <div style={{ fontSize: 14, fontFamily: "sans-serif" }}>No inference history yet</div>
              <div style={{ fontSize: 12, color: "#30363d", marginTop: 6, fontFamily: "sans-serif" }}>
                Run your first analysis from the Dashboard tab
              </div>
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: "1rem", alignItems: "start" }}>

              {/* History list */}
              <div style={card}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: ".875rem" }}>
                  <div style={secTitle}>
                    <div style={titleIcon}>🕓</div>
                    History ({history.length})
                  </div>
                  <button
                    onClick={() => { setHistory([]); setSelectedHistory(null); setPredicted(null); setDetections([]); }}
                    style={{ fontSize: 11, color: "#ef4444", background: "transparent",
                      border: "none", cursor: "pointer", fontFamily: "sans-serif" }}
                  >
                    Clear all
                  </button>
                </div>
                {history.map((item) => (
                  <HistoryItem
                    key={item.id}
                    item={item}
                    onSelect={loadFromHistory}
                    isSelected={selectedHistory === item.id}
                  />
                ))}
              </div>

              {/* Selected history detail */}
              <div>
                {selectedHistory && predicted ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <div style={card}>
                      <div style={secTitle}>
                        <div style={titleIcon}>◈</div>
                        {history.find(h => h.id === selectedHistory)?.filename}
                      </div>
                      <img src={predicted} alt="history prediction"
                        style={{ width: "100%", borderRadius: 8, border: "1px solid #21262d" }} />
                      <div style={{ display: "flex", gap: 8, marginTop: ".75rem", flexWrap: "wrap" }}>
                        <div style={chip("#e6edf3", "#21262d")}>{detections.length} detections</div>
                        <div style={chip("#3fb950", "#122d1f")}>✓ {inferTime}ms</div>
                      </div>
                    </div>

                    <div style={card}>
                      <div style={secTitle}>
                        <div style={titleIcon}>◈</div>
                        Detections
                      </div>
                      {detections.length === 0 ? (
                        <div style={{ fontSize: 12, color: "#484f58", fontFamily: "sans-serif" }}>
                          No damages detected in this image
                        </div>
                      ) : (
                        detections.map((d, i) => {
                          const col  = CLASS_COLORS[d.class]?.bg   || "#ef4444";
                          const icon = CLASS_COLORS[d.class]?.icon || "◆";
                          return (
                            <div key={i} style={{
                              display: "flex", alignItems: "center", gap: 8, padding: "8px 0",
                              borderBottom: i < detections.length - 1 ? "1px solid #21262d" : "none",
                            }}>
                              <div style={{
                                width: 34, height: 34, background: col + "22",
                                border: `1px solid ${col}44`, borderRadius: 7,
                                display: "flex", alignItems: "center", justifyContent: "center",
                                fontSize: 14, color: col, flexShrink: 0,
                              }}>
                                {icon}
                              </div>
                              <div>
                                <div style={{ fontSize: 13, fontWeight: 600, fontFamily: "sans-serif",
                                  color: "#e6edf3", textTransform: "capitalize" }}>
                                  {d.class.charAt(0).toUpperCase() + d.class.slice(1)}
                                </div>
                                <div style={{ fontSize: 10, color: "#8b949e", fontFamily: "sans-serif" }}>
                                  Confidence: {(d.confidence * 100).toFixed(1)}%
                                  &nbsp;·&nbsp;BBox: [{d.bbox.join(", ")}]
                                </div>
                              </div>
                              <div style={{
                                marginLeft: "auto", padding: "2px 8px", borderRadius: 5,
                                fontSize: 11, fontWeight: 700, fontFamily: "monospace",
                                background: confColor(d.confidence), color: "#fff", flexShrink: 0,
                              }}>
                                {d.confidence.toFixed(2)}
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                ) : (
                  <div style={{ ...card, textAlign: "center", color: "#484f58", padding: "3rem" }}>
                    <div style={{ fontSize: 32, marginBottom: 8 }}>←</div>
                    <div style={{ fontSize: 12, fontFamily: "sans-serif" }}>Select an item from history</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Model tab ───────────────────────────────────────── */}
      {tab === "model" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", maxWidth: 800 }}>
            <div style={card}>
              <div style={secTitle}><div style={titleIcon}>◈</div>Architecture</div>
              {[
                ["Model",     "YOLOv8m"],
                ["Classes",   "6"],
                ["Input size","640×640"],
                ["Optimizer", "SGD"],
                ["lr0",       "0.01"],
                ["Epochs",    "100"],
                ["Patience",  "20"],
              ].map(([k, v]) => (
                <div key={k} style={{
                  display: "flex", justifyContent: "space-between", fontSize: 12,
                  padding: "6px 0", borderBottom: "1px solid #21262d",
                  fontFamily: "sans-serif", color: "#8b949e",
                }}>
                  <span>{k}</span>
                  <span style={{ color: "#e6edf3", fontWeight: 600, fontFamily: "monospace" }}>{v}</span>
                </div>
              ))}
            </div>
            <div style={card}>
              <div style={secTitle}><div style={titleIcon}>◉</div>Metrics</div>
              {[
                ["mAP@50",    "87.4%"],
                ["mAP@50-95", "62.1%"],
                ["Precision", "88.2%"],
                ["Recall",    "83.1%"],
                ["F1-Score",  "85.6%"],
                ["mIoU",      "76.2%"],
              ].map(([k, v]) => (
                <div key={k} style={{
                  display: "flex", justifyContent: "space-between", fontSize: 12,
                  padding: "6px 0", borderBottom: "1px solid #21262d",
                  fontFamily: "sans-serif", color: "#8b949e",
                }}>
                  <span>{k}</span>
                  <span style={{ color: "#3fb950", fontWeight: 600, fontFamily: "monospace" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}