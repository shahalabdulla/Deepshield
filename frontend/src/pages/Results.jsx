import { useLocation, useNavigate } from "react-router-dom";

function Results() {
  const navigate = useNavigate();
  const location = useLocation();
  const { state } = location || {};

  const verdict    = state?.verdict    || "FAKE";
  const confidence = state?.confidence ?? 0.91;
  const filename   = state?.filename   || "sample-video.mp4";
  const heatmapUrl = state?.heatmap_url || null;

  const models = state?.models || [
    { name: "Xception",     score: 0.94 },
    { name: "EfficientNet", score: 0.88 },
    { name: "MesoNet",      score: 0.90 },
  ];

  const getVerdictStyle = () => {
    if (verdict === "REAL")      return "real";
    if (verdict === "UNCERTAIN") return "uncertain";
    return "fake";
  };

  const ringClass =
    verdict === "REAL"
      ? "score-ring score-ring-real"
      : verdict === "FAKE"
      ? "score-ring score-ring-fake"
      : "score-ring";

      const handleDownloadReport = async () => {
        try {
          const reportData = {
            filename: filename,
            verdict: verdict,
            confidence: (confidence * 100).toFixed(1),
            xception_score: (models[0]?.score * 100 || 0).toFixed(1),
            efficientnet_score: (models[1]?.score * 100 || 0).toFixed(1),
            mesonet_score: (models[2]?.score * 100 || 0).toFixed(1),
            frames_analyzed: state?.frames_analyzed || 0,
            processing_time: state?.processing_time || 0,
            heatmap_url: heatmapUrl,
          }
    
          const res = await fetch("http://127.0.0.1:8000/report-from-data", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reportData)
          })
    
          if (!res.ok) throw new Error("Report generation failed")
    
          const blob = await res.blob()
          const url = URL.createObjectURL(blob)
          const link = document.createElement("a")
          link.href = url
          link.download = `deepshield-report-${filename}.pdf`
          link.click()
          URL.revokeObjectURL(url)
    
        } catch (err) {
          console.error(err)
          alert("Could not generate report. Make sure backend is running.")
        }
      };
  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <nav className="navbar">
        <div className="logo">DeepShield</div>
        <div className="flex gap-md">
          <button className="btn btn-ghost btn-md" onClick={() => navigate("/")}>
            Landing
          </button>
          <button
            className="btn btn-secondary btn-md"
            onClick={() => navigate("/upload")}
          >
            Analyze another file
          </button>
        </div>
      </nav>

      <main
        className="container"
        style={{
          flex: 1,
          paddingTop: "var(--space-3xl)",
          paddingBottom: "var(--space-4xl)",
        }}
      >
        <header
          style={{
            marginBottom: "var(--space-2xl)",
            display: "flex",
            justifyContent: "space-between",
            gap: "var(--space-xl)",
            flexWrap: "wrap",
          }}
        >
          <div>
            <h1
              style={{
                fontSize: "clamp(28px, 4vw, 40px)",
                fontWeight: 900,
                marginBottom: "var(--space-sm)",
              }}
            >
              Analysis results
            </h1>
            <p className="text-secondary text-sm">
              File: <span className="text-primary">{filename}</span>
            </p>
          </div>
          <div
            className="flex gap-md"
            style={{ alignItems: "center", justifyContent: "flex-end", flexWrap: "wrap" }}
          >
            <button
              className="btn btn-secondary btn-md"
              onClick={() => navigate("/upload")}
            >
              New analysis
            </button>
            <button className="btn btn-primary btn-md" onClick={handleDownloadReport}>
              Download report
            </button>
          </div>
        </header>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 2fr) minmax(0, 3fr)",
            gap: "var(--space-xl)",
          }}
        >
          {/* ── Left column ── */}
          <section>
            <div className="card-elevated" style={{ marginBottom: "var(--space-xl)" }}>
              <div
                className="flex"
                style={{ gap: "var(--space-xl)", alignItems: "center", flexWrap: "wrap" }}
              >
                <div className={ringClass}>
                  <div
                    className="text-xs text-muted"
                    style={{ textTransform: "uppercase", letterSpacing: "0.08em" }}
                  >
                    Confidence
                  </div>
                  <div className="font-black" style={{ fontSize: "var(--font-3xl)" }}>
                    {(confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted" style={{ marginTop: 4 }}>
                    model estimate
                  </div>
                </div>

                <div style={{ minWidth: 220 }}>
                  <div
                    className={`badge badge-${getVerdictStyle()}`}
                    style={{ marginBottom: "var(--space-sm)" }}
                  >
                    Verdict: {verdict}
                  </div>
                  <p className="text-secondary text-sm">
                    This verdict is generated by aggregating outputs from multiple
                    deep learning models. Use it as decision support, not as a sole
                    source of truth.
                  </p>
                </div>
              </div>
            </div>

            <div className="card">
              <h2 className="font-semibold text-lg" style={{ marginBottom: "var(--space-md)" }}>
                Model scores
              </h2>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                  gap: "var(--space-md)",
                }}
              >
                {models.map((m) => (
                  <div
                    key={m.name}
                    style={{
                      borderRadius: "var(--radius-md)",
                      border: "1px solid var(--border)",
                      padding: "var(--space-md)",
                      background: "rgba(10,10,15,0.6)",
                    }}
                  >
                    <div className="font-semibold text-sm" style={{ marginBottom: 4 }}>
                      {m.name}
                    </div>
                    <div className="text-secondary text-xs" style={{ marginBottom: 8 }}>
                      Deepfake likelihood
                    </div>
                    <div className="progress-bar" style={{ marginBottom: 6 }}>
                      <div
                        className="progress-fill"
                        style={{ width: `${(m.score * 100).toFixed(0)}%` }}
                      />
                    </div>
                    <div className="text-muted text-xs">
                      {(m.score * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ── Right column ── */}
          <section>
            <div className="card" style={{ marginBottom: "var(--space-xl)" }}>
              <h2 className="font-semibold text-lg" style={{ marginBottom: "var(--space-md)" }}>
                Heatmap visualization
              </h2>

              {/* ── Real heatmap OR placeholder ── */}
              <div
                style={{
                  borderRadius: "var(--radius-lg)",
                  border: "1px solid var(--border)",
                  background: "#050509",
                  overflow: "hidden",
                  marginBottom: "var(--space-md)",
                  minHeight: 260,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {heatmapUrl ? (
                  <img
                    src={heatmapUrl}
                    alt="Grad-CAM heatmap"
                    style={{
                      width: "100%",
                      display: "block",
                      borderRadius: "var(--radius-lg)",
                    }}
                  />
                ) : (
                  <div
                    style={{
                      position: "relative",
                      width: "100%",
                      height: 260,
                      overflow: "hidden",
                    }}
                  >
                    {/* Decorative gradient placeholder */}
                    <div
                      style={{
                        position: "absolute",
                        inset: 0,
                        background:
                          "radial-gradient(circle at 30% 40%, rgba(239,68,68,0.4), transparent 60%), radial-gradient(circle at 70% 60%, rgba(99,102,241,0.35), transparent 55%)",
                        opacity: 0.9,
                      }}
                    />
                    <div
                      style={{
                        position: "absolute",
                        inset: 0,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: "var(--space-sm)",
                      }}
                    >
                      <div className="text-secondary text-sm">
                        Heatmap not available for this file
                      </div>
                      <div className="text-muted text-xs">
                        Heatmap is generated for image files only
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <p className="text-secondary text-sm">
                {heatmapUrl
                  ? "Red regions indicate areas most likely to be AI-manipulated. Generated using Grad-CAM on the Xception model."
                  : "The heatmap highlights regions that contributed most to the deepfake likelihood estimate. Brighter areas indicate higher contribution."}
              </p>
            </div>

            <div className="card">
              <h2 className="font-semibold text-lg" style={{ marginBottom: "var(--space-md)" }}>
                How to interpret these results
              </h2>
              <ul
                className="text-secondary text-sm"
                style={{ paddingLeft: "1.1rem", display: "grid", gap: "0.35rem" }}
              >
                <li>
                  High confidence does not guarantee that a sample is authentic or
                  manipulated; it reflects the model estimate given the observed patterns.
                </li>
                <li>
                  Review heatmaps and model scores together, especially for
                  borderline or high-risk decisions.
                </li>
                <li>
                  Consider running multiple samples from the same source to
                  identify systematic issues.
                </li>
                <li>
                  Always pair automated analysis with human review in critical or
                  legal contexts.
                </li>
              </ul>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

export default Results;