import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";

function Upload() {
  const navigate = useNavigate();
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const handleFileSelect = (selected) => {
    if (!selected || !selected[0]) return;
    const f = selected[0];
    const allowed = ["video/", "image/"];
    if (!allowed.some((prefix) => f.type.startsWith(prefix))) {
      setError("Only video and image files are supported.");
      return;
    }
    setError("");
    setFile(f);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    handleFileSelect(e.dataTransfer.files);
  }, []);

  const onBrowseChange = (e) => {
    handleFileSelect(e.target.files);
  };

  const startUpload = async () => {
    if (!file || uploading) return;
    setUploading(true);
    setError("");
    setProgress(10);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const interval = setInterval(() => {
        setProgress((p) => (p >= 85 ? p : p + 8));
      }, 400);

      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      clearInterval(interval);

      if (!response.ok) throw new Error("Analysis failed");

      const result = await response.json();
      setProgress(100);

      await new Promise((res) => setTimeout(res, 400));

      // ── Navigate with heatmap_url included ──
      navigate("/results", {
        state: {
          filename: file.name,
          size: file.size,
          verdict: result.verdict,
          confidence: result.confidence / 100,
          models: [
            { name: "Xception",     score: result.xception_score / 100 },
            { name: "EfficientNet", score: result.efficientnet_score / 100 },
            { name: "MesoNet",      score: result.mesonet_score / 100 },
          ],
          frames_analyzed: result.frames_analyzed,
          processing_time: result.processing_time,
          heatmap_url: result.heatmap_url,   // ← real heatmap from backend
        },
      });
    } catch (err) {
      console.error(err);
      setError(
        "Could not reach analysis server. Make sure the backend is running."
      );
      setUploading(false);
      setProgress(0);
    }
  };

  const cancelUpload = () => {
    setUploading(false);
    setProgress(0);
    setFile(null);
    setError("");
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <nav className="navbar">
        <div className="logo">DeepShield</div>
        <div className="flex gap-md">
          <button className="btn btn-ghost btn-md" onClick={() => navigate("/")}>
            Landing
          </button>
        </div>
      </nav>

      <main
        className="container-sm"
        style={{
          flex: 1,
          paddingTop: "var(--space-4xl)",
          paddingBottom: "var(--space-4xl)",
        }}
      >
        <header style={{ marginBottom: "var(--space-2xl)", textAlign: "center" }}>
          <h1
            style={{
              fontSize: "clamp(32px, 5vw, 48px)",
              fontWeight: 900,
              marginBottom: "var(--space-sm)",
            }}
          >
            Upload media for analysis
          </h1>
          <p className="text-secondary text-base">
            Supported formats: MP4, MOV, MKV, JPEG, PNG.
            Files are deleted immediately after analysis.
          </p>
        </header>

        <section>
          <div
            className={`upload-zone${dragging ? " dragging" : ""}`}
            onDragEnter={(e) => { e.preventDefault(); e.stopPropagation(); setDragging(true); }}
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
            onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setDragging(false); }}
            onDrop={onDrop}
            onClick={() => {
              if (uploading) return;
              document.getElementById("file-input")?.click();
            }}
          >
            <input
              id="file-input"
              type="file"
              style={{ display: "none" }}
              onChange={onBrowseChange}
              accept="video/*,image/*"
            />

            <h2 className="font-semibold text-xl" style={{ marginBottom: "var(--space-sm)" }}>
              Drag and drop a file here
            </h2>
            <p className="text-secondary text-sm" style={{ marginBottom: "var(--space-md)" }}>
              or click to browse from your device.
            </p>

            {file && (
              <div
                className="card"
                style={{
                  marginTop: "var(--space-md)",
                  textAlign: "left",
                  maxWidth: 520,
                  marginInline: "auto",
                }}
              >
                <div className="text-sm text-secondary">Selected file</div>
                <div className="font-semibold" style={{ marginTop: "var(--space-xs)" }}>
                  {file.name}
                </div>
                <div className="text-muted text-xs">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </div>
              </div>
            )}
          </div>

          {error && (
            <p
              className="text-sm"
              style={{ color: "#fca5a5", marginTop: "var(--space-md)", textAlign: "center" }}
            >
              {error}
            </p>
          )}

          {uploading && (
            <div style={{ marginTop: "var(--space-xl)" }}>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <div
                className="text-secondary text-sm"
                style={{ marginTop: "var(--space-sm)", textAlign: "right" }}
              >
                {progress}% — running 3 model analysis...
              </div>
            </div>
          )}

          <div
            className="flex gap-md"
            style={{ marginTop: "var(--space-2xl)", justifyContent: "flex-end" }}
          >
            <button
              className="btn btn-secondary btn-md"
              disabled={!uploading}
              onClick={cancelUpload}
              style={uploading ? {} : { opacity: 0.4, cursor: "not-allowed" }}
            >
              Cancel
            </button>
            <button
              className="btn btn-primary btn-md"
              disabled={!file || uploading}
              onClick={startUpload}
              style={!file || uploading ? { opacity: 0.4, cursor: "not-allowed" } : {}}
            >
              Start analysis
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Upload;