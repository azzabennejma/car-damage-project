import "./user.css";
import { useState, useRef, useEffect } from "react";
import { FiLogOut } from "react-icons/fi";
import { FiClock, FiCpu } from "react-icons/fi";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { LayoutDashboard  } from "lucide-react";
import { Bell } from "lucide-react";
import { Camera } from "lucide-react";
import { Database } from "lucide-react";
import { History } from "lucide-react";
import { ChartNoAxesCombined  } from "lucide-react";




import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";


// ── DonutChart ───────────────────────────────────────────────
function DonutChart({ value }) {
  const r = 38, cx = 50, cy = 50;
  const circ = 2 * Math.PI * r;

  const segments = [
    { pct: value / 100,  color: "#e57a1d" },
    
  ];

  let cumulative = 0;

  return (
    <svg width={100} height={100} viewBox="0 0 100 100">
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke="#21262d"
        strokeWidth={9}
      />

      {segments.map((seg, i) => {
        const dash = circ * seg.pct;
        const gap = circ - dash;

        const el = (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={seg.color}
            strokeWidth={9}
            strokeDasharray={`${dash} ${gap}`}
            strokeDashoffset={-(cumulative * circ)}
            transform={`rotate(-90 ${cx} ${cy})`}
            strokeLinecap="butt"
          />
        );

        cumulative += seg.pct;
        return el;
      })}

      <text
        x={cx}
        y={cy - 7}
        textAnchor="middle"
        fill="#15232f"
        fontSize={15}
        fontWeight={700}
      >
        mAP
      </text>

      <text
        x={cx}
        y={cy + 14}
        textAnchor="middle"
        fill="#15232f"
        fontSize={15}
        fontWeight={700}
      >
        {value}%
      </text>
    </svg>
  );
}

export default function User() {
  // ───── STATE ─────
  const [accuracyHistory, setAccuracyHistory] = useState([
    { batch: "v0", accuracy: 74.4 },
    { batch: "v1", accuracy: 80.2 },
    { batch: "v2", accuracy: 78.6 }
  ]);
  
  const [currentBatchAcc, setCurrentBatchAcc] = useState([]);
  const [datasetVersions, setDatasetVersions] = useState([]);
  const [modelPerformance, setModelPerformance] = useState([]);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictionCount, setPredictionCount] = useState(0);

  const [serverOk, setServerOk] = useState(null);
  const API_BASE = "http://localhost:8000";

  const [detections, setDetections] = useState([]);
  const [history, setHistory] = useState([]);

  const [kpis, setKpis] = useState(null);
  const [model, setModel] = useState(null);

  const [step, setStep] = useState("upload"); // "upload" | "result"
  const [conf, setConf] = useState(0.25);

  const [tab, setTab] = useState("dashboard");

  const [logins, setLogins] = useState([]);
  const [datasetQueue, setDatasetQueue] = useState([]);

  const fileRef = useRef();
  const [loading, setLoading] = useState(false);
  const [error, setError]= useState(null);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const lastNotificationId = useRef(null);
  const datasetInputRef = useRef();
  const [modelStatus, setModelStatus] = useState("deployed");
  
  const [stats, setStats] = useState(null);

  const username = localStorage.getItem("username");
  const role = localStorage.getItem("role");
  const formData = new FormData();
  formData.append("file", file);
  formData.append("username", localStorage.getItem("username"));
  formData.append("conf", 0.25);


  // ───── FASTAPI CALL: INFERENCE ─────
  const runInference = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    
    // Send logged-in username
    formData.append(
      "username",
      localStorage.getItem("username")
    );

    formData.append("file", file);
    formData.append("conf",  0.25 );
    
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error("Prediction failed");
      }

      const data = await res.json();
      setDetections(data.detections);
      const avgConf =
        data.detections.length > 0
          ? data.detections.reduce(
              (sum, d) => sum + d.confidence,
              0
            ) / data.detections.length
          : 0;

      

         // 👈 THIS FIXES YOUR UI
      setPredictionCount(prev => prev + 1);

      setCurrentBatchAcc(prev => {
        const updated = [...prev, avgConf * 100];

        if (updated.length === 10) {
          const batchAvg =
            updated.reduce((a, b) => a + b, 0) / 10;

          setAccuracyHistory(hist => [
            ...hist,
            {
              batch: `${accuracyHistory.length * 10 + 1}-${(accuracyHistory.length + 1) * 10}`,
              accuracy: Number(batchAvg.toFixed(1))
            }
          ]);

          return []; // reset after 10 predictions
        }

        return updated;
      });
      // convert base64 → image
      const imgUrl = `data:image/jpeg;base64,${data.image_base64}`;

      setPreview(imgUrl);

      setStep("result"); // 👈 SWITCH VIEW HERE

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  const card = {
  background: "rgba(79, 196, 255, 0.32)",
  border: "1px solid rgba(99, 128, 157, 0.737)",
  
  borderRadius: 12,
  padding: "1rem",
};

  const secTitle = {
    fontSize: 13,
    fontWeight: 700,
    color: "#e6edf3",
    marginBottom: ".875rem",
  };

  const titleIcon = {
    width: 26,
    height: 26,
    background: "#042e3c",
    borderRadius: 6,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  };

  
  const fetchStats = async () => {
    try {
      const username = localStorage.getItem("username");

      const res = await fetch(`http://localhost:8000/user/stats/${username}`);
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.log("stats error", err);
    }
  };
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
  const uploadDataset = async (e) => {

  const files = e.target.files;

  if (!files.length) return;

  const username =
    localStorage.getItem("username");

  const version =
    prompt("Dataset version? (v1, v2, v3...)");

  if (!version) return;

  const formData = new FormData();

  formData.append(
    "username",
    username
  );

  formData.append(
    "version",
    version
  );

  for (let file of files) {
    formData.append(
      "files",
      file
    );
  }

  const res = await fetch(
      "http://localhost:8000/upload-dataset",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await res.json();

    alert(
      `${data.files} files uploaded to ${data.version}`
    );
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(
          "http://localhost:8000/notifications"
        );

        const data = await res.json();

        setNotifications(data);

        if (data.length === 0) return;

        // First dashboard load: store ID silently
        if (lastNotificationId.current === null) {
          lastNotificationId.current = data[0].id;
          return;
        }

        // Only show truly new notifications
        if (data[0].id !== lastNotificationId.current) {
          toast.info(data[0].message);

          setUnreadCount(c => c + 1);

          lastNotificationId.current = data[0].id;
        }

      } catch (err) {
        console.error(err);
      }
    }, 500);

    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    fetch("http://localhost:8000/users/logins")
      .then(res => res.json())
      .then(data => setLogins(data))
      .catch(err => console.error(err));
  }, []);

  useEffect(() => {
    fetch("http://localhost:8000/dataset-queue")
      .then(res => res.json())
      .then(data => setDatasetQueue(data))
      .catch(err => console.error(err));
  }, []);
  useEffect(() => {

    const interval = setInterval(async () => {

      const res = await fetch(
        "http://localhost:8000/pipeline-status"
      );

      const data = await res.json();

      setPipeline(data);

    }, 3000);

    return () => clearInterval(interval);

  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetch("http://localhost:8000/dataset-history")
      .then(res => res.json())
      .then(data => setDatasetVersions(data))
      .catch(err => console.error(err));
  }, []);
  useEffect(() => {
    fetch("/api/model-status")
      .then(res => res.json())
      .then(data => setModelStatus(data.status));
  }, []);
  // ───── FILE HANDLER ─────
  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;

    setFile(f);
    setPreview(URL.createObjectURL(f));
    setStep("upload"); // still upload mode
  };
  const combinedHistory = [
    ...logins.map(l => ({
      event: "Login",
      user: l.username,
      details: "Successful login",
      time: `${l.date} ${l.time}`
    })),

    ...datasetQueue.map(d => ({
      event: "Inference",
      user: d.username,
      details: `${d.num_detections} detections (${d.infer_ms} ms)`,
      time: d.created_at
    }))
  ].sort(
    (a, b) => new Date(b.time) - new Date(a.time)
  );

  // ───── NAVIGATION RENDER ─────
  const renderContent = () => {
    switch (tab) {
      case "dashboard":
        return (
          <>
            {/* KPI */}
            <div className="kpi-grid">
              <div className="kpi analyses">
                <h3>Total Analyses</h3>
                <h3>{stats?.total_analyses ?? 0}</h3>

                <div className="kpi-chart">
                  <svg viewBox="0 0 100 40">
                    <polyline
                      fill="none"
                      stroke="#22c55e"
                      strokeWidth="4"
                      points="0,30 20,20 40,25 60,10 80,18 100,8"
                    />
                    {[
                      [0,30],[20,20],[40,25],[60,10],[80,18],[100,8]
                    ].map(([x,y], i) => (
                      <circle key={i} cx={x} cy={y} r="2" fill="#22c55e" />
                    ))}
                  </svg>
                </div>
              </div>
              
              <div className="kpi detections">
                <h3>Total Detections</h3>
                <h3>{stats?.total_detections ?? 0}</h3>

                <div className="kpi-chart">
                  <svg viewBox="0 0 100 40">
                    <polyline
                      fill="none"
                      stroke="#f97316"
                      strokeWidth="4"
                      points="0,30 20,20 40,25 60,10 80,18 100,8"
                    />
                    {[
                     [0,30],[20,20],[40,25],[60,10],[80,18],[100,8]
                    ].map(([x,y], i) => (
                      <circle key={i} cx={x} cy={y} r="2" fill="#f97316" />
                    ))}
                  </svg>
                </div>
              </div>

              <div className="kpi model">
                <h3>Model</h3>
                <p>{model?.model_version ?? "v2.0"}</p>

                <img
                  src="/model.png"
                  alt="AI Model"
                  className="kpi-illustration"
                />
              </div>

              <div className="kpi">
              <h3>Last Inference</h3>
              <p>{stats?.last_inference ?? "..."}</p>

              <FiClock className="kpi-icon clock-icon" />
              </div>           
            </div>
            
            {/* UPLOAD + RESULT */}
            
            <div className="grid-2">
              {/* UPLOAD CARD */}
              {step === "upload" && (
                <div className="card upload">
            
                  <div
                    className="dropzone"
                    onClick={() => fileRef.current.click()}
                  >
                    <input
                      ref={fileRef}
                      type="file"
                      hidden
                      onChange={handleFile}
                    />

                    {preview ? (
                      <img src={preview} alt="preview" />
                    ) : (
                      <p>Drag & Drop or Click</p>
                    )}
                  </div>

                  <button className="runBtn" onClick={runInference}>
                    {loading ? "Processing..." : "Run Inference"}
                  </button>
                </div>
              )}

              {/* RESULT CARD */}
              {step === "result" && (
                <div className="card result">
                    <h3>Prediction Result</h3>

                    <div className="result-grid">

                      {/* IMAGE */}
                      <div className="result-image">
                        {preview ? (
                          <img src={preview} alt="result" />
                        ) : (
                          <p>No image</p>
                        )}
                      </div>

                      {/* DETECTIONS */}
                      <div className="result-detections">
                        {detections.map((d, i) => (
                          <div
                            key={i}
                            className={`tag ${d.class.toLowerCase()}`}
                          >
                            <strong>{d.class}</strong>
                            <span>
                              {(d.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>

                    </div>
                  

                  <button
                    className="runBtn"
                    onClick={() => {
                      setStep("upload");
                      setFile(null);
                      setPreview(null);
                      setDetections([]);
                    }}
                  >
                    New Prediction
                  </button>
                </div>
              )}
              {/* Model Performance */}
              <div style={card}>
                <div style={secTitle}>
                  
                  <h3>Recent Training Results</h3>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: ".875rem", marginTop: ".5rem" }}>
                <DonutChart value={78.6} />
                  <div style={{ flex: 1 }}>
                    {[
                      ["Precision", "80.1%"],
                      ["Recall",    "73.4%"],
                      /*["F1-Score",  "85.6%"],*/
                    ].map(([label, val, col]) => (
                      <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 5 }}>
                        <div style={{ width: 7, height: 7, borderRadius: "50%", background: "rgba(46, 84, 103, 0.57)", flexShrink: 0 }} />
                        <span style={{ fontSize: 11, color: "#7598a6", fontFamily: "sans-serif", flex: 1 }}>{label}:</span>
                        <span style={{ fontSize: 11, fontWeight: 600, fontFamily: "sans-serif", color: "#e6edf3" }}>{val}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div
                  style={{
                    height: 160,
                    marginTop: "1rem",
                    borderTop: "1px solid #21262d",
                    paddingTop: "1rem",
                  }}
                >
                  <h3>Model Performance (mAP50%)</h3>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={accuracyHistory}>
                      <XAxis dataKey="batch" stroke="#ffffff" />
                      <YAxis domain={[50, 100]} ticks={[50, 74.4, 90, 100]} stroke="#ffffff" />
                      <Tooltip />
                      <Line
                        type="linear"
                        dataKey="accuracy"
                        stroke="#f6ff00"
                        strokeWidth={1}
                        dot={{ r: 3, fill: "#ffffff" }}
                        activeDot={{ r: 8 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            
            </div>

          </>
        );

      case "prediction":
        return (
          <div className="grid-2">

            <div className="card upload">
              <h3>Run Inference</h3>

              <div
                className="dropzone"
                onClick={() => fileRef.current.click()}
              >
                <input
                  ref={fileRef}
                  type="file"
                  hidden
                  onChange={handleFile}
                />

                {preview ? (
                  <img src={preview} alt="preview" />
                ) : (
                  <p>Click to upload image</p>
                )}
              </div>

              <button
                className="runBtn"
                onClick={runInference}
                disabled={!file}
              >
                {loading ? "Processing..." : "Run Inference"}
              </button>
            </div>

            <div className="card result">
              <h3>Prediction Results</h3>

              {preview ? (
                <img src={preview} alt="result" />
              ) : (
                <p>No result yet</p>
              )}

              <div style={{ marginTop: 15 }}>
                {detections.map((d, i) => (
                  <div key={i}>
                    <strong>{d.class}</strong>
                    {" "}
                    ({(d.confidence * 100).toFixed(1)}%)
                  </div>
                ))}
              </div>
            </div>

          </div>
          
        );
      case "Tracking":
        return (
          <div className="card">
            <h3>Experiment Tracking</h3>

            <p>
              Access MLflow to inspect experiments,
              metrics, artifacts, and model versions.
            </p>

            <button
              className="runBtn"
              onClick={() =>
                window.open(
                  "http://localhost:5000",
                  "_blank"
                )
              }
            >
              Open MLflow
            </button>
          </div>
        );  

      case "history":
        return (
          <div className="card">
            <h2>Activity History</h2>

            <table className="db-table">
              <thead>
                <tr>
                  <th style={{ width: "20%" }}>User</th>
                  <th style={{ width: "15%" }}>Event</th>
                  <th style={{ width: "20%" }}>Time</th>
                  <th style={{ width: "45%" }}>Details</th>
                </tr>
              </thead>

              <tbody>
                {combinedHistory.map((h, i) => (
                  <tr key={i}>
                    <td>{h.user}</td>

                    <td>
                      {h.event === "Login" ? " Login" : " Inference"}
                    </td>

                    <td>{h.time}</td>

                    <td>{h.details}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      
      case "datasets":
        return (
          <div className="card">
            <h2>Dataset Versions</h2>

            <table className="db-table">
              <thead>
                <tr>
                  <th>Version</th>
                  <th>User</th>
                  <th>Files</th>
                  <th>Date</th>
                  <th>Time</th>
                </tr>
              </thead>

              <tbody>
                {datasetVersions.map((d, i) => (
                  <tr key={i}>
                    <td>{d.version}</td>
                    <td>{d.username}</td>
                    <td>{d.num_files}</td>
                    <td>{d.date}</td>
                    <td>{d.time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        
        

      case "notifications":
        return (
          <div className="card">
            <h2>Notifications</h2>

            <table className="db-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Message</th>
                  <th>Created At</th>
                </tr>
              </thead>

              <tbody>
                {notifications.map((n) => (
                  <tr key={n.id}>
                    <td>{n.id}</td>
                    <td>{n.message}</td>
                    <td>{n.created_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
    }
  };

  // ───── UI ─────
  return (
    <>
      <ToastContainer
        position="top-right"
        autoClose={5000}
        theme="dark"
      />
    
    <div className="dashboard">
      

      {/* SIDEBAR */}
      <div className="sidebar">
        <div className="logo">
          <img src="background.png" alt="Damage Detect Logo" />
        </div>

        <div className="menu">
          <button className={tab === "dashboard" ? "active" : ""} onClick={() => setTab("dashboard")}><LayoutDashboard />Dashboard</button>
          <button className={tab === "" ? "active" : ""} onClick={() => setTab("prediction")}><Camera />Prediction</button>
          <button className={tab === "datasets" ? "active" : ""} onClick={() => setTab("datasets")}><Database />Datasets</button>
          <button className={tab === "Tracking" ? "active" : ""} onClick={() => setTab("Tracking")}><ChartNoAxesCombined />Tracking</button>
          <button className={tab === "history" ? "active" : ""} onClick={() => setTab("history")}><History />History</button>
          <button className={tab === "notifications" ? "active" : ""} onClick={() => setTab("notifications")}><Bell />Notifications</button>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            borderTop: "1px solid #333",
            paddingTop: 10,
          }}
        >
          
        </div>
        <div className="bottom">
          {/* Server status */}
          <div
            onClick={checkServer}
            title="Click to ping server"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              cursor: "pointer",
              marginBottom: 12,
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background:
                  serverOk === null
                    ? "#999"
                    : serverOk
                    ? "#22c55e"
                    : "#ef4444",
              }}
            />
            <span style={{ fontSize: 12, opacity: 0.8 }}>
              {serverOk === null
                ? "Check server"
                : serverOk
                ? "API online"
                : "API offline"}
            </span>
          </div>
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
              background: "#73b1ce",
              color: "#000",
              cursor: "pointer",
              fontWeight: 600,
              marginBottom: 12,
            }}

          >
            <FiLogOut size={18} />
            Log Out
          </button>
          <div className="profile">
            <img
              className="avatar"
              src="https://cdn-icons-png.flaticon.com/512/149/149071.png"
              alt="anonymous avatar"
            />
            <div>
              <p className="name">
                Welcome {username || "User"}
              </p>
              
            </div>
          </div>
          <img
            src="/seca.png"
            alt="logo"
            className="enterprise-logo"
          />
        </div>

      </div>

      {/* MAIN */}
      <div className="main">
        <div className="header">
          <div>
            <h2>Welcome Back !</h2>
            <p>Analyze vehicle damage</p>
          </div>

          <div
            className="notif"
            onClick={() => {
              setTab("notifications");
              setUnreadCount(0);
            }}
          >
            🔔

            {unreadCount > 0 && (
              <span className="notif-badge">
                {unreadCount}
              </span>
            )}
          </div>
          
        </div>

        {renderContent()}
      </div>

    </div>
    </>
  );
}