import { useNavigate } from "react-router-dom";

function Landing() {
  const navigate = useNavigate();

  const features = [
    {
      title: "Multi-model pipeline",
      desc: "Xception, EfficientNet and MesoNet are combined for robust deepfake detection.",
    },
    {
      title: "Heatmap evidence",
      desc: "Spatial overlays highlight regions that most influenced the model decision.",
    },
    {
      title: "Forensic reports",
      desc: "Export structured PDF reports suitable for investigative and legal workflows.",
    },
    {
      title: "Low latency",
      desc: "Optimized inference delivers results in under five seconds on typical content.",
    },
    {
      title: "Security first",
      desc: "Short-lived processing, encrypted transport, and no long-term media storage.",
    },
    {
      title: "Global coverage",
      desc: "Language-agnostic models for content from any region or platform.",
    },
  ];

  const stats = [
    { number: "97%", label: "Detection accuracy*" },
    { number: "3", label: "Specialized AI models" },
    { number: "<5s", label: "Median analysis time" },
    { number: "24/7", label: "Continuous availability" },
  ];

  const steps = [
    {
      title: "Upload your media",
      desc: "Drag and drop a video or image, or select a file from your device. Files are processed transiently.",
      badge: "Step 1",
    },
    {
      title: "AI-driven analysis",
      desc: "Multiple models inspect spatial and temporal patterns to estimate manipulation likelihood.",
      badge: "Step 2",
    },
    {
      title: "Review and export",
      desc: "Inspect verdicts, confidence scores and heatmaps, then export a detailed report if needed.",
      badge: "Step 3",
    },
  ];

  const trustPoints = [
    "Designed for analysts and investigators",
    "Encrypted transport and isolated processing",
    "No training on uploaded customer content",
    "Backed by research-grade architectures",
  ];

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <nav className="navbar">
        <div className="logo">DeepShield</div>
        <div className="flex gap-md">
          <button
            className="btn btn-ghost btn-md"
            onClick={() => {
              const el = document.getElementById("how-it-works");
              if (el) el.scrollIntoView({ behavior: "smooth" });
            }}
          >
            How it works
          </button>
          <button className="btn btn-ghost btn-md">Docs</button>
          <button
            className="btn btn-primary btn-md"
            onClick={() => navigate("/upload")}
          >
            Upload video
          </button>
        </div>
      </nav>

      <main style={{ flex: 1 }}>
        <section
          className="container-sm text-center animate-fade-in"
          style={{
            paddingTop: "var(--space-4xl)",
            paddingBottom: "var(--space-3xl)",
          }}
        >
          <div
            className="badge badge-primary"
            style={{ marginBottom: "var(--space-xl)" }}
          >
            AI-powered deepfake detection for high-stakes environments
          </div>

          <h1
            style={{
              fontSize: "clamp(40px, 8vw, 72px)",
              fontWeight: 900,
              lineHeight: 1.1,
              marginBottom: "var(--space-lg)",
              letterSpacing: "-1px",
            }}
          >
            Verify video authenticity{" "}
            <span className="gradient-text">in seconds</span>.
          </h1>

          <p
            className="text-secondary text-xl"
            style={{
              marginBottom: "var(--space-2xl)",
              maxWidth: "640px",
              margin: "0 auto var(--space-2xl)",
            }}
          >
            DeepShield analyzes videos and images with multiple neural networks to
            detect manipulation, highlight suspicious regions, and generate
            evidence-ready reports — without retaining your media.
          </p>

          <div className="flex gap-md justify-center">
            <button
              className="btn btn-primary btn-xl"
              onClick={() => navigate("/upload")}
            >
              Start analysis
            </button>
            <button
              className="btn btn-secondary btn-xl"
              onClick={() => {
                const el = document.getElementById("how-it-works");
                if (el) el.scrollIntoView({ behavior: "smooth" });
              }}
            >
              View workflow
            </button>
          </div>

          <p
            className="text-muted text-sm"
            style={{ marginTop: "var(--space-md)" }}
          >
            *Representative accuracy based on internal benchmarking. Performance
            may vary by dataset.
          </p>
        </section>

        <section
          style={{
            borderTop: "1px solid var(--border)",
            borderBottom: "1px solid var(--border)",
            padding: "var(--space-2xl) 0",
            marginBottom: "var(--space-3xl)",
          }}
        >
          <div
            className="container flex justify-center gap-xl"
            style={{ flexWrap: "wrap" }}
          >
            {stats.map((s) => (
              <div key={s.label} className="text-center">
                <div
                  className="gradient-text font-black"
                  style={{ fontSize: "var(--font-4xl)" }}
                >
                  {s.number}
                </div>
                <div
                  className="text-muted text-sm"
                  style={{ marginTop: "var(--space-xs)" }}
                >
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section
          className="container"
          style={{ marginBottom: "var(--space-4xl)" }}
        >
          <h2
            className="text-center font-black text-3xl"
            style={{ marginBottom: "var(--space-2xl)" }}
          >
            Why organizations choose{" "}
            <span className="gradient-text">DeepShield</span>
          </h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: "var(--space-lg)",
            }}
          >
            {features.map((f) => (
              <div key={f.title} className="card text-center">
                <h3
                  className="font-bold text-lg"
                  style={{ marginBottom: "var(--space-sm)" }}
                >
                  {f.title}
                </h3>
                <p className="text-secondary text-sm">{f.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section
          id="how-it-works"
          className="container"
          style={{ marginBottom: "var(--space-4xl)" }}
        >
          <h2
            className="text-center font-black text-3xl"
            style={{ marginBottom: "var(--space-2xl)" }}
          >
            How DeepShield works
          </h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
              gap: "var(--space-lg)",
            }}
          >
            {steps.map((step) => (
              <div key={step.title} className="card">
                <div
                  className="badge badge-primary"
                  style={{ marginBottom: "var(--space-sm)" }}
                >
                  {step.badge}
                </div>
                <h3
                  className="font-semibold text-lg"
                  style={{ marginBottom: "var(--space-sm)" }}
                >
                  {step.title}
                </h3>
                <p className="text-secondary text-sm">{step.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section
          className="container"
          style={{ marginBottom: "var(--space-4xl)" }}
        >
          <div className="card-primary">
            <div
              className="flex justify-between items-center"
              style={{ gap: "var(--space-xl)", flexWrap: "wrap" }}
            >
              <div style={{ minWidth: 260 }}>
                <h3
                  className="font-bold text-2xl"
                  style={{ marginBottom: "var(--space-sm)" }}
                >
                  Built for high-stakes decisions
                </h3>
                <p className="text-secondary text-sm">
                  DeepShield is designed for teams that need defensible, explainable
                  analysis of digital media.
                </p>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                  gap: "var(--space-md)",
                  flex: 1,
                }}
              >
                {trustPoints.map((item) => (
                  <div
                    key={item}
                    className="text-sm text-secondary"
                    style={{
                      padding: "var(--space-sm) var(--space-md)",
                      borderRadius: "var(--radius-md)",
                      border: "1px solid var(--border)",
                      background: "rgba(10,10,15,0.6)",
                    }}
                  >
                    {item}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer
        style={{
          background: "var(--gradient-glow)",
          borderTop: "1px solid var(--border-primary)",
          padding: "var(--space-4xl) var(--space-xl)",
          textAlign: "center",
        }}
      >
        <h2
          className="font-black"
          style={{
            fontSize: "clamp(28px, 5vw, 44px)",
            marginBottom: "var(--space-md)",
          }}
        >
          Ready to assess your content?
        </h2>
        <p
          className="text-secondary text-lg"
          style={{ marginBottom: "var(--space-xl)" }}
        >
          Begin with a single video or image. No account required.
        </p>
        <button
          className="btn btn-primary btn-xl"
          onClick={() => navigate("/upload")}
        >
          Launch DeepShield
        </button>
      </footer>
    </div>
  );
}

export default Landing;

