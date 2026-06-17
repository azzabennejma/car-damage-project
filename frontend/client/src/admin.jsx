import "./Admin.css";

import { useState, useRef, useCallback, useEffect } from "react";
import { FiLogOut } from "react-icons/fi";
import { AiOutlineLoading3Quarters } from "react-icons/ai";
import { FaPlay } from "react-icons/fa";
import { data } from "react-router-dom";
import { FiDatabase } from "react-icons/fi";
import { FiActivity } from "react-icons/fi";
import { FiCpu } from "react-icons/fi";
// ── Config ──────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

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
    { pct: value / 100,               color: "#3c1b04" },
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
        background: isSelected ? "#0a658422" : "transparent",
        border: isSelected ? "1px solid #0a6f8e55" : "1px solid transparent",
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
export default function Admin() {
  const [file, setFile]             = useState(null);
  const [preview, setPreview]       = useState(null);
  const [predicted, setPredicted]   = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading]       = useState(false);
  const [inferTime, setInferTime]   = useState(null);
  const [conf, setConf]             = useState(0.25);
  const [dragging, setDragging]     = useState(false);
  const [tab, setTab]               = useState("overview");
  const [error, setError]           = useState(null);
  const [serverOk, setServerOk]     = useState(null);
  const [history, setHistory]       = useState([]);       // ← stores all past inferences
  const [selectedHistory, setSelectedHistory] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState("Idle");
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const inputRef = useRef();
  const [mlflowMetrics, setMlflowMetrics] = useState(null);
  const [mlflow, setMlflow] = useState(null);
  const [users, setUsers] = useState([]);
  const [stats, setStats] = useState(null);
  
  const canRetrain = stats?.can_retrain;
  const threshold = stats?.threshold || 200;

  useEffect(() => {
  fetch("http://localhost:8000/system/stats")
    .then(res => res.json())
    .then(data => setStats(data));
}, []);

  // 👇 ADD THIS HERE (RIGHT AFTER STATES)
  useEffect(() => {
    fetchStats();
  }, []);
  useEffect(() => {
    fetchHistory();
  }, []);
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const res = await fetch("http://localhost:8000/users/logins");
        const data = await res.json();
        setUsers(data);
      } catch (err) {
        console.log(err);
      }
    };

    fetchUsers();
  }, []);

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
  const login = async () => {
    const res = await fetch("http://localhost:8000/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username,
        password,
      }),
    });

    const data = await res.json();

    if (!res.ok) throw new Error(data.detail);

    localStorage.setItem("user", JSON.stringify(data));
  };
  const register = async () => {
    const res = await fetch("http://localhost:8000/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username,
        email,
        password,
      }),
    });

    const data = await res.json();

    if (!res.ok) throw new Error(data.detail);

    return data;
  };
  const fetchMetrics = async () => {
    const res = await fetch("http://localhost:8000/mlflow/latest");
    const data = await res.json();
    setMetrics(data.metrics);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);
  const fetchStats = async () => {
    const res = await fetch("http://localhost:8000/stats");
    const data = await res.json();

    setStats(data);
  };
  const fetchHistory = async () => {
    try {
      const res = await fetch("http://localhost:8000/history");
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      console.log(err);
    }
  };


  // ── Server health check ───────────────────────────────────
  const checkServer = async () => {
    try {
      const res = await fetch("http://localhost:8000/health");

      if (!res.ok) {
        setServerOk(false);
        return;
      }

      const data = await res.json();

      setServerOk(data.status === "ok");
    } catch (err) {
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
  const triggerRetrain = async () => {
  try {
    setPipelineRunning(true);
    setPipelineStatus("Starting pipeline...");

    const res = await fetch(`${API_BASE}/retrain`, {
      method: "POST",
    });

    const data = await res.json();

    setPipelineStatus(data.message);
  } catch (err) {
    setPipelineStatus("Failed");
  }

  setPipelineRunning(false);
};

  // ── Shared styles ─────────────────────────────────────────
  const card = {
    background: "", border: "1px solid #21262d",
    borderRadius: 10, padding: "1rem",
  };
  const secTitle = {
    fontSize: 13, fontWeight: 700, color: "#e6edf3",
    marginBottom: ".875rem", display: "flex", alignItems: "center",
    gap: 7, fontFamily: "sans-serif",
  };
  const titleIcon = {
    width: 26, height: 26, background: "#042e3c", borderRadius: 6,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 12, flexShrink: 0,
  };
  const chip = (col, bg) => ({
    padding: "4px 12px", borderRadius: 16, fontSize: 11,
    fontWeight: 600, fontFamily: "sans-serif", background: bg, color: col,
  });

  // ── Render ────────────────────────────────────────────────
  return (
    <div 
      style={{

        minHeight: "100vh",
        display: "flex",
        backgroundImage: "url('/car4.png')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        color: "#fff",
        fontFamily: "sans-serif",
      }}
    >


        {/* SIDEBAR */}
    <div
      style={{
        width: "240px",
        borderRight: "1px solid #ffffff",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
      }}
    >
      <div style={{ fontSize: 18, fontWeight: "bold", color: "#31a2bb" }}>
         Car Damage Detection
      </div>

      {[
        { id: "overview", label: "Overview" },
        { id: "History", label: "History" },
        { id: "dataset", label: "Dataset Queue" },
        { id: "pipeline", label: "Pipeline" },
        { id: "experiments", label: "Experiments" },
        { id: "registry", label: "Model Registry" },
        { id: "users", label: "Users" },
        { id: "settings", label: "Settings" },
      ].map((t) => (
        <button
          key={t.id}
          onClick={() => setTab(t.id)}
          style={{
            padding: "10px",
            borderRadius: "8px",
            border: "none",
            cursor: "pointer",
            textAlign: "left",
            background: tab === t.id ? "#31a2bb" : "transparent",
            color: tab === t.id ? "#000" : "#fff",
            fontWeight: 600,
          }}
        >
          {t.label}
        </button>
      ))}
      <div style={{ flex: 1, padding: "20px" }}>
        {/* ALL YOUR EXISTING TABS CONTENT HERE */}
      </div>

      {/* Server status */}
        <div onClick={checkServer} title="Click to ping server"
          style={{ display: "flex", alignItems: "center", gap: 5, cursor: "pointer", marginLeft: 4 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: serverOk === null ? "#ffffff" : serverOk ? "#3fb950" : "#ef4444",
          }} />
          <span style={{ fontSize: 11, color: "#ffffff", fontFamily: "sans-serif" }}>
            {serverOk === null ? "Check server" : serverOk ? "API online" : "API offline"}
          </span>
        </div>
        {/* Logout Button */}
        <button
          onClick={() => {
            localStorage.removeItem("token");
            window.location.href = "/";
          }}
          style={{
            width: "100%",
            padding: "10px",
            borderRadius: "8px",
            border: "none",
            background: "#4dbcef",
            color: "#000000",
            cursor: "pointer",
            fontWeight: 600,
            marginBottom: 20,
          }}
        >
          <FiLogOut size={18} />
          Log Out
        </button>

        {/* Enterprise Logo */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            borderTop: "1px solid #484f58",
            paddingTop: 15,
          }}
        >
          <img
            src="/seca.png"
            alt="Company Logo"
            style={{
              width: 170,
              height: 40,
              objectFit: "contain",
              filter: "drop-shadow(0 0 8px rgba(245, 247, 248, 0.76))",
            }}
          />

        </div>
      

    </div>



      {/* ── Dashboard tab ──────────────────────────────────── */}
      {tab === "overview" && (
        <div style={{ padding: "1rem", flex: 1 }}>

          {/* OVERVIEW STATS */}
         <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "14px",
              marginBottom: "1rem",
            }}
          >
            {/* Box 1 */}
            <div
            style={{
              ...card,
              minHeight: "120px",
              borderRadius: "20px",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              padding: "16px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              position: "relative",
              overflow: "hidden",
            }}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: "11px",
                    fontWeight: 900,
                    color: "#ffffff",
                    letterSpacing: "0.5px",

                  }}
                >
                  DATASET QUEUE
                </div>

                <div
                  style={{
                    fontSize: "30px",
                    fontWeight: "800",
                    color: "#fff",
                    marginTop: "4px",
                  }}
                >
                  {stats?.pending_datasets ?? 5}
                </div>
              </div>

              {/* Icon */}
              <div
                style={{
                  width: 46,
                  height: 46,
                  borderRadius: "14px",
                  background:
                    "linear-gradient(135deg, rgba(49,162,187,0.15), rgba(255,140,0,0.15))",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <FiDatabase
                  size={22}
                  color="#31a2bb"
                />
              </div>
            </div>

            {/* Status */}
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                background: "rgba(255,140,0,0.10)",
                border: "1px solid rgba(255,140,0,0.2)",
                borderRadius: "999px",
                padding: "4px 10px",
                width: "fit-content",
                fontSize: "11px",
                color: "#ffb347",
                marginTop: "12px",
              }}
            >
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: "#ff9800",
                }}
              />
              Pending Validation
            </div>
          </div>

            {/* Box 2 */}
            <div
              style={{
                ...card,
                minHeight: "120px",
                borderRadius: "20px",
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                padding: "16px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {/* Header */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <div
                    style={{
                      fontSize: "11px",
                      fontWeight: 800,
                      color: "#ffffff",
                      letterSpacing: "0.5px",
                    }}
                  >
                    TRAINING RUNS
                  </div>

                  <div
                    style={{
                      fontSize: "30px",
                      fontWeight: "800",
                      color: "#fff",
                      marginTop: "4px",
                    }}
                  >
                    {stats?.training_runs ?? 3}
                  </div>
                </div>

                {/* Icon */}
                <div
                  style={{
                    width: 46,
                    height: 46,
                    borderRadius: "14px",
                    background:
                      "linear-gradient(135deg, rgba(49,162,187,0.18), rgba(255,140,0,0.10))",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <FiActivity size={22} color="#31a2bb" />
                </div>
              </div>

              {/* Status Pill */}
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  background: "rgba(49,162,187,0.10)",
                  border: "1px solid rgba(49,162,187,0.25)",
                  borderRadius: "999px",
                  padding: "4px 10px",
                  width: "fit-content",
                  fontSize: "11px",
                  color: "#31a2bb",
                  marginTop: "12px",
                }}
              >
                <div
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#31a2bb",
                    animation: "pulse 1.5s infinite",
                  }}
                />
                In Progress
              </div>
            </div>
            {/* Box 3 */}
            <div
              style={{
                ...card,
                minHeight: "120px",
                borderRadius: "20px",
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                padding: "16px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {/* Header */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <div
                    style={{
                      fontSize: "11px",
                      fontWeight: 800,
                      color: "#ffffff",
                      letterSpacing: "0.5px",
                    }}
                  >
                    MODEL REGISTRY
                  </div>

                  <div
                    style={{
                      fontSize: "30px",
                      fontWeight: "800",
                      color: "#fff",
                      marginTop: "4px",
                    }}
                  >
                    {stats?.models_trained ?? 3}
                  </div>
                </div>

                {/* Icon */}
                <div
                  style={{
                    width: 46,
                    height: 46,
                    borderRadius: "14px",
                    background:
                      "linear-gradient(135deg, rgba(49,162,187,0.18), rgba(255,1400,0,0.10))",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <FiCpu size={22} color="#31a2bb" />
                </div>
              </div>

              {/* Status */}
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  background: "rgba(255,140,0,0.10)",
                  border: "1px solid rgba(255,140,0,0.25)",
                  borderRadius: "999px",
                  padding: "4px 10px",
                  width: "fit-content",
                  fontSize: "11px",
                  color: "#ffb347",
                  marginTop: "12px",
                }}
              >
                <div
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#ffb347",
                  }}
                />
                Versioned Models
              </div>
            </div>
            {/* Model Version Box */}
            <div
              style={{
                ...card,
                minHeight: "120px",
                borderRadius: "20px",
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                
                padding: "16px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {/* Top */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <div
                    style={{
                      fontSize: "11px",
                      fontWeight: 800,
                      color: "#fcfcfc",
                      letterSpacing: "0.5px",
                    }}
                  >
                    ACTIVE MODEL
                  </div>

                  <div
                    style={{
                      fontSize: "30px",
                      fontWeight: "800",
                      color: "#fff",
                      marginTop: "4px",
                    }}
                  >
                    V{stats?.model_version || "2.1"}
                  </div>
                </div>

                {/* AI Symbol */}
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: "14px",
                    background:
                      "linear-gradient(135deg, rgba(49,162,187,0.15), rgba(255,140,0,0.15))",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "20px",
                  }}
                >
                  ⬢
                </div>
              </div>

              {/* Status Badge */}
              <div
                style={{
                  marginTop: "10px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  background: "rgba(49,162,187,0.10)",
                  border: "1px solid rgba(49,162,187,0.2)",
                  borderRadius: "999px",
                  padding: "4px 10px",
                  width: "fit-content",
                  fontSize: "11px",
                  color: "#31a2bb",
                }}
              >
                <div
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#31a2bb",
                  }}
                />
                Production
              </div>

            {/* Version Switcher */}
            <div
              style={{
                display: "flex",
                gap: "8px",
                marginTop: "14px",
              }}
            >
              {["1.0", "2.0", "3.0"].map((v) => (
                <button
                  key={v}
                  onClick={() =>
                    setStats((prev) => ({
                      ...prev,
                      model_version: v,
                    }))
                  }
                  style={{
                    flex: 1,
                    padding: "8px",
                    borderRadius: "12px",
                    border:
                      stats?.model_version === v
                        ? "1px solid #31a2bb"
                        : "1px solid rgba(255,255,255,0.08)",
                    background:
                      stats?.model_version === v
                        ? "rgba(49,162,187,0.12)"
                        : "transparent",
                    color:
                      stats?.model_version === v
                        ? "#31a2bb"
                        : "#c9d1d9",
                    fontSize: "11px",
                    fontWeight: "600",
                    cursor: "pointer",
                    transition: "all .2s ease",
                  }}
                >
                  V{v}
                </button>
              ))}
            </div>
          </div>
          </div>

          {/* MAIN GRID */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "260px 1fr 260px",
              gap: "1rem",
              alignItems: "start",
            }}
          >

            {/* LEFT — Upload */}
            <div
              style={{
                ...card,
                background: "rgba(79, 196, 255, 0.32)",
                backdropFilter: "blur(1.5px)",
                  
              }}
              
            >
              <div style={secTitle}>
                <div style={titleIcon}>⬆</div>
                Upload Image
              </div>

              <div
                onClick={() => inputRef.current.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                style={{
                  border: `2px dashed ${dragging ? "#31a2bb" : "#30363d"}`,
                  borderRadius: 8,
                  padding: "1.5rem 1rem",
                  textAlign: "center",
                  cursor: "pointer",
                  marginBottom: ".75rem",
                  background: dragging
                    ? "rgba(79, 196, 255, 0.32)"
                    : "transparent",
                  transition: "all .2s",
                }}
              >
                <div style={{ fontSize: 28, color: "#348abc", marginBottom: 6 }}>
                  ☁
                </div>
                <div style={{ fontSize: 12, color: "#fff" }}>
                  Drag & drop your image here
                </div>
                <div style={{ fontSize: 11, color: "#fff" }}>
                  or click to browse
                </div>

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    inputRef.current.click();
                  }}
                  style={{
                    background: "#3486bc",
                    color: "#fff",
                    border: "none",
                    borderRadius: 7,
                    padding: "8px 16px",
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: "pointer",
                    marginTop: ".75rem",
                  }}
                >
                  ⬆ Choose Image
                </button>

                <div style={{ fontSize: 10, color: "#fff", marginTop: 6 }}>
                  JPG, PNG up to 10MB
                </div>
              </div>

              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])}
              />

              {/* Preview */}
              {preview && (
                <div style={{ marginBottom: ".625rem" }}>
                  <div style={{ fontSize: 10, fontWeight: 600, color: "#fff" }}>
                    Preview
                  </div>
                  <img
                    src={preview}
                    alt="preview"
                    style={{
                      width: "100%",
                      borderRadius: 7,
                      maxHeight: 150,
                      objectFit: "cover",
                    }}
                  />
                </div>
              )}

              {/* Run button */}
              <button
                disabled={!file || loading}
                onClick={runInference}
                style={{
                  width: "100%",
                  padding: "10px 0",
                  fontSize: 13,
                  fontWeight: 600,
                  borderRadius: 7,
                  border: "none",
                  cursor: file && !loading ? "pointer" : "not-allowed",
                  background: file && !loading ? "#31a2bb" : "#21262d",
                  color: "#fff",
                }}
              >
                {loading ? "Processing..." : "Analyze Image"}
              </button>

              {error && (
                <div
                  style={{
                    background: "#31a2bb",
                    borderRadius: 8,
                    padding: ".75rem 1rem",
                    fontSize: 11,
                    marginTop: ".75rem",
                    whiteSpace: "pre-line",
                  }}
                >
                  <strong>Error:</strong> {error}
                </div>
              )}
            </div>

            {/* CENTER — Prediction */}
            <div
              style={{
                 background: "rgba(79, 196, 255, 0.32)",
                border: "1px solid rgba(49,162,187,0.3)",
                borderRadius: 16,
                padding: 16,
                display: "flex",
                flexDirection: "column",
                gap: 14,
              }}
            >

              {/* HEADER */}
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <div style={{ fontSize: 14, fontWeight: 900, color: "#ffffff" }}>
                  📊 DATASET PROGRESS
                </div>

                <div
                  style={{
                    fontSize: 11,
                    padding: "4px 10px",
                    borderRadius: 20,
                    background: stats?.can_retrain
                      ? "rgba(186, 8, 8, 0.15)"
                      : "rgba(49,162,187,0.15)",
                    color: stats?.can_retrain ? "#ffb4b4" : "#ecf4f6",
                    fontWeight: 800,
                  }}
                >
                  {stats?.can_retrain ? "🚨 ACTION REQUIRED" : "🟦 STABLE"}
                </div>
              </div>

              {/* BIG NUMBER */}
              <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                <div style={{ fontSize: 44, fontWeight: 900, color: "#fff" }}>
                  {stats?.total_analyses ?? 0}
                </div>
                <div style={{ fontSize: 12,fontWeight: 800, color: "#8b949e" }}>
                  / {stats?.threshold}
                </div>
              </div>

              {/* PROGRESS BAR */}
              <div
                style={{
                  width: "100%",
                  height: 14,
                  background: "#21262d",
                  borderRadius: 20,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${Math.min(
                      ((stats?.total_analyses || 0) / 40) * 100,
                      100
                    )}%`,
                    background: stats?.can_retrain ? "#ff4d4d" : "#31a2bb",
                    transition: "0.5s ease",
                  }}
                />
              </div>

              {/* STATUS TEXT */}
              <div style={{ fontSize: 12,fontWeight: 800, color: "#ffffff" }}>
                {stats?.can_retrain
                  ? "🚨 Dataset threshold reached — retraining recommended"
                  : "📡 Collecting data... model learning in progress"}
              </div>

              {/* ACTIONS */}
              <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
                
                {/* CHECK DATA */}
                <button
                  onClick={fetchHistory}
                  style={{
                    flex: 1,
                    padding: "10px",
                    borderRadius: 10,
                    border: "1px solid rgba(49,162,187,0.4)",
                    background: "rgba(49,162,187,0.90)",
                    color: "#ffffff",
                    fontWeight: 900,
                    cursor: "pointer",
                  }}
                >
                  📁 Data
                </button>

                

                {/* IGNORE */}
                <button
                  onClick={() =>
                    setStats((prev) => ({ ...prev, can_retrain: false }))
                  }
                  style={{
                    flex: 1,
                    padding: "10px",
                    borderRadius: 10,
                    border: "1px solid rgba(255, 0, 0, 0.4)",
                    background: "rgba(242, 35, 35, 0.81)",
                    color: "#fff6f6",
                    fontWeight: 900,
                    cursor: "pointer",
                  }}
                >
                  ❌ Ignore
                </button>

              </div>
            </div>
            {/* RIGHT — MODEL METRICS */}
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>

              {/* MAIN CARD */}
              <div
                style={{
                  ...card,
                   background: "rgba(79, 196, 255, 0.32)",
                  border: "1px solid rgba(49,162,187,0.3)",
                  borderRadius: 16,
                  padding: 16,
                }}
              >

                {/* TITLE */}
                <div style={secTitle}>
                  <div style={titleIcon}>📊</div>
                  Last Model Performance
                </div>

                {/* ARC + ACCURACY */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    marginTop: 10,
                  }}
                >
                  <svg width="140" height="140" viewBox="0 0 100 100">
                    {/* background circle */}
                    <circle
                      cx="50"
                      cy="50"
                      r="38"
                      fill="none"
                      stroke="#21262d"
                      strokeWidth="10"
                    />

                    {/* accuracy arc */}
                    <circle
                      cx="50"
                      cy="50"
                      r="38"
                      fill="none"
                      stroke="#c25100"
                      strokeWidth="10"
                      strokeDasharray={`${2 * Math.PI * 38}`}
                      strokeDashoffset={`${2 * Math.PI * 38 * 0.22}`} // 78% example
                      strokeLinecap="round"
                      transform="rotate(-90 50 50)"
                    />

                    {/* center text */}
                    <text
                      x="50"
                      y="48"
                      textAnchor="middle"
                      fill="#ffffff"
                      fontSize="13"
                      fontWeight="700"
                    >
                      Accuracy
                    </text>

                    <text
                      x="50"
                      y="65"
                      textAnchor="middle"
                      fill="#c25100"
                      fontSize="18"
                      fontWeight="900"
                    >
                      78%
                    </text>
                  </svg>
                </div>

                {/* MODEL NAME */}
                <div
                  style={{
                    textAlign: "center",
                    fontSize: 12,
                    color: "#f5f5f5",
                    marginTop: 6,
                    fontWeight: 800,
                  }}
                >
                  YOLOv8m — last trained model
                </div>
              </div>

              {/* METRICS ROW */}
              <div
                style={{
                  ...card,
                   background: "rgba(79, 196, 255, 0.32)",
                  border: "1px solid rgba(49,162,187,0.2)",
                  borderRadius: 16,
                  padding: 14,
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 10,
                }}
              >

                {/* PRECISION */}
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11,fontWeight: 900, color: "#ffffff" }}>Precision</div>
                  <div style={{ fontSize: 20, fontWeight: 900, color: "#c25100" }}>
                    86.4%
                  </div>
                </div>

                {/* RECALL */}
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11,fontWeight: 900, color: "#ffffff" }}>Recall</div>
                  <div style={{ fontSize: 20, fontWeight: 900, color: "#c25100" }}>
                    81.7%
                  </div>
                </div>

                {/* F1 */}
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11,fontWeight: 900, color: "#ffffff" }}>F1 Score</div>
                  <div style={{ fontSize: 20, fontWeight: 900, color: "#c25100" }}>
                    84.0%
                  </div>
                </div>

                {/* mAP */}
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11,fontWeight: 900, color: "#ffffff" }}>mAP</div>
                  <div style={{ fontSize: 20, fontWeight: 900, color: "#c25100" }}>
                    87.4%
                  </div>
                </div>

              </div>
            </div>

          </div> {/* end grid */}

        </div>
      )} {/* end dashboard */}




      {/* ── History tab ─────────────────────────────────────── */}
      {tab === "history" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          {history.length === 0 ? (
            <div style={{ textAlign: "center", color: "#ffffff", paddingTop: "4rem" }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>🕓</div>
              <div style={{ fontSize: 14, fontFamily: "sans-serif" }}>No inference history yet</div>
              <div style={{ fontSize: 12, color: "#30363d", marginTop: 6, fontFamily: "sans-serif" }}>
                Run your first analysis from the Dashboard tab
              </div>
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: "1rem", alignItems: "start" }}>

              {/* History list */}
              <div
                style={{
                  ...card,
                  background: "rgba(79, 196, 255, 0.32)",
                  backdropFilter: "blur(1.5px)",
                }}
              >
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
                  <div key={item.id} style={card}>
                    <div style={{ fontSize: 12 }}>
                      📄 {item.filename}
                    </div>

                    <div style={{ fontSize: 11 }}>
                      🔍 Detections: {item.num_detections}
                    </div>

                    <div style={{ fontSize: 11 }}>
                      ⏱ {item.infer_ms} ms
                    </div>
                  </div>
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

                    <div
                      style={{
                        ...card,
                        background: "rgba(79, 196, 255, 0.32)",
                        backdropFilter: "blur(1.5px)",
                      }}
                    >
                      <div style={secTitle}>
                        <div style={titleIcon}>◈</div>
                        Detections
                      </div>
                      {detections.length === 0 ? (
                        <div style={{ fontSize: 12, color: "#ffffff", fontFamily: "sans-serif" }}>
                          No damages detected in this image
                        </div>
                      ) : (
                        detections.map((d, i) => {
                          const col  = CLASS_COLORS[d.class]?.bg   || "#ef4444";
                          const icon = CLASS_COLORS[d.class]?.icon || "◆";
                          return (
                            <div key={i} style={{
                              display: "flex", alignItems: "center", gap: 8, padding: "8px 0",
                              borderBottom: i < detections.length - 1 ? "1px solid #bc7834" : "none",
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
                                  &nbsp;·&nbsp;BBox: [{d.bbox?.join(", ") || "N/A"}]
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
                  <div style={{ ...card, textAlign: "center", color: "#ffffff", padding: "3rem" }}>
                    <div style={{ fontSize: 32, marginBottom: 8 }}>←</div>
                    <div style={{ fontSize: 12, fontFamily: "sans-serif" }}>Select an item from history</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
      {tab === "dataset" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          <div style={card}>
            <div style={secTitle}>
              <div style={titleIcon}>📦</div>
              Dataset Queue
            </div>

            <table style={{ width: "100%", fontSize: 12, color: "#fff" }}>
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Uploaded By</th>
                  <th>Date</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>v1.0</td>
                  <td>Admin</td>
                  <td>12/06/26</td>
                  <td>14:22</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}


      {/* ── Model tab ───────────────────────────────────────── */}
      {tab === "model" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", maxWidth: 800 }}>
            <div 
              style={{
                ...card,
                background: "rgba(79, 196, 255, 0.32)",
                backdropFilter: "blur(1.5px)",
              }}
            >
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
                  padding: "6px 0", borderBottom: "1px solid #000000bb",
                  fontFamily: "sans-serif", color: "#8b949e",
                }}>
                  <span>{k}</span>
                  <span style={{ color: "#e6edf3", fontWeight: 600, fontFamily: "monospace" }}>{v}</span>
                </div>
              ))}
            </div>
            <div
              style={{
                ...card,
                background: "rgba(79, 196, 255, 0.32)",
                backdropFilter: "blur(1.5px)",
              }}
            >
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
      {tab === "pipeline" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          <div style={card}>
            <div style={secTitle}>
              <div style={titleIcon}>⚙</div>
              Training Pipeline
            </div>

            {[
              "Dataset Validation",
              "Data Versioning",
              "Training",
              "Evaluation",
              "Registration",
              "Deployment",
            ].map((step, i) => (
              <div key={i} style={{
                padding: "10px",
                marginBottom: 6,
                background: i === 2 ? "#f97316" : "#21262d",
                borderRadius: 6,
                color: "#fff"
              }}>
                {i < 2 ? "✓" : i === 2 ? "⚡" : "○"} {step}
              </div>
            ))}
          </div>
        </div>
      )}
      {tab === "experiments" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          <div style={card}>
            <div style={secTitle}>
              <div style={titleIcon}>🧪</div>
              Experiments
            </div>

            <table style={{ width: "100%", fontSize: 12, color: "#fff" }}>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Model</th>
                  <th>Precision</th>
                  <th>Recall</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>EXP-001</td>
                  <td>YOLOv8m</td>
                  <td>88.2%</td>
                  <td>83.1%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
      {tab === "registry" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          <div style={card}>
            <div style={secTitle}>
              <div style={titleIcon}>📚</div>
              Model Registry
            </div>

            <table style={{ width: "100%", fontSize: 12, color: "#fff" }}>
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Stage</th>
                  <th>mAP</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>v2.1</td>
                  <td>Production</td>
                  <td>85.6%</td>
                  <td>
                    <button>Promote</button>
                    <button>Archive</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
      {tab === "users" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          <div style={card}>
            <div style={secTitle}>
              <div style={titleIcon}>👥</div>
              Active Users
            </div>

            <table style={{ width: "100%", fontSize: 12, color: "#fff" }}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Date</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u, i) => (
                  <tr key={i}>
                    <td>{u.username}</td>
                    <td>{u.date}</td>
                    <td>{u.time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {tab === "settings" && (
        <div style={{ padding: "1rem", flex: 1 }}>
          <div style={card}>
            <div style={secTitle}>
              <div style={titleIcon}>⚙</div>
              Settings
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <label><input type="checkbox" /> Dark Mode</label>
              <label><input type="checkbox" /> Auto Refresh</label>
              <label><input type="checkbox" /> Notifications</label>

              <button
                onClick={() => {
                  localStorage.removeItem("token");
                  window.location.href = "/";
                }}
                style={{
                  marginTop: 10,
                  background: "#ef4444",
                  color: "#fff",
                  padding: 10,
                  borderRadius: 8,
                  border: "none"
                }}
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      )}

     

    </div>
  );
}