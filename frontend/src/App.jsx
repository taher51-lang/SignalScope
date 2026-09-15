import { useState, useCallback } from "react";
import "./App.css";

const API_BASE = "https://taher52-signalscope.hf.space";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [caption, setCaption] = useState("");

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      setError("Please upload a valid image file (JPEG, PNG, or WebP).");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("File is too large. Please upload an image under 20MB.");
      return;
    }
    setError(null);
    setResult(null);
    setShowHeatmap(false);
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragActive(false);
  }, []);

  const handleSubmit = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", image);

    const hasCaption = caption.trim().length > 0;
    const endpoint = hasCaption
      ? `${API_BASE}/predict_multimodal`
      : `${API_BASE}/predict`;

    if (hasCaption) formData.append("caption", caption.trim());

    try {
      const res = await fetch(endpoint, { method: "POST", body: formData });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Prediction failed. Please try again.");
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Network error. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setImage(null);
    setPreview(null);
    setResult(null);
    setShowHeatmap(false);
    setCaption("");
    setError(null);
  };

  const isFake = result?.label === "AI-GENERATED";
  const isInconclusive = result?.label === "INCONCLUSIVE";
  const verdictClass = isInconclusive ? "is-inconclusive" : (isFake ? "is-fake" : "is-real");

  return (
    <>
      <header className="app-header">
        <div className="app-logo">
          <div className="logo-mark">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12h4l3-9 5 18 3-9h5" />
            </svg>
          </div>
          <span className="logo-text">SignalScope</span>
        </div>
        <span className="header-tag">Internal Hackathon</span>
      </header>

      <main className="app-main">
        <section className="hero">
          <h1>Image Authenticity Analysis</h1>
          <p>
            Upload an image to determine if it's a real photograph or AI-generated.
            Review the metadata, visual explanation, and caption consistency.
          </p>
        </section>

        {!preview && (
          <div
            className={`dropzone ${dragActive ? "active" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById("fileInput").click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                document.getElementById("fileInput").click();
              }
            }}
          >
            <div className="dropzone-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="dropzone-title">Click or drag image to upload</p>
            <p className="dropzone-sub">Maximum file size 20MB.</p>
            <div className="dropzone-formats">
              <span className="fmt">JPEG</span>
              <span className="fmt">PNG</span>
              <span className="fmt">WEBP</span>
            </div>
            <input
              id="fileInput"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="file-input"
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>
        )}

        {preview && (
          <div className="card">
            <div className="card-image">
              <div className="card-image-inner">
                <img
                  src={showHeatmap && result?.heatmap ? result.heatmap : preview}
                  alt={showHeatmap ? "Explanation Heatmap" : "Uploaded Image"}
                />
              </div>
              {result?.heatmap && (
                <div className="view-toggle">
                  <button
                    className={`view-btn ${!showHeatmap ? "on" : ""}`}
                    onClick={() => setShowHeatmap(false)}
                  >
                    Original
                  </button>
                  <button
                    className={`view-btn ${showHeatmap ? "on" : ""}`}
                    onClick={() => setShowHeatmap(true)}
                  >
                    Heatmap
                  </button>
                </div>
              )}
            </div>

            {!result && !loading && (
              <div className="card-controls">
                <input
                  type="text"
                  className="caption-field"
                  placeholder="Add a caption to check consistency (optional)"
                  value={caption}
                  onChange={(e) => setCaption(e.target.value)}
                />
                <p className="caption-hint">
                  Computes semantic similarity between image and text.
                </p>
                <div className="btn-row">
                  <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
                    Run Analysis
                  </button>
                  <button className="btn-ghost" onClick={reset}>
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {loading && (
              <div className="loading-area">
                <div className="spinner"></div>
                <p>Analyzing image...</p>
                <p>Running CLIP ViT-B/32 and metadata extraction</p>
              </div>
            )}

            {result && (
              <div className="result">
                <div className={`verdict ${verdictClass}`}>
                  <div className="verdict-dot"></div>
                  <div className="verdict-body">
                    <div className="verdict-label">
                      {isInconclusive ? "Inconclusive / Not Sure" : (isFake ? "Likely AI-Generated" : "Likely Real")}
                    </div>
                    <div className="verdict-sub">
                      {isInconclusive ? "Confidence too borderline to classify (0.45-0.55)" : "Based on visual artifacts and model threshold (0.50)"}
                    </div>
                  </div>
                  <div className="verdict-pct">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div className={`conf-bar ${verdictClass}`}>
                  <div className="conf-track">
                    <div
                      className="conf-fill"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>

                <div className="signals">
                  <div className="sig">
                    <div className="sig-label">Provenance & Metadata</div>

                    {/* C2PA Check */}
                    {result.metadata?.has_c2pa ? (
                      <div style={{ marginBottom: 12 }}>
                        <div className="sig-value text-green-600">Content Credentials Found</div>
                        <div className="sig-detail">
                          <span className="tag tag-green">C2PA Standard</span>
                        </div>
                      </div>
                    ) : (
                      <div style={{ marginBottom: 12 }}>
                        <div className="sig-value text-tertiary">No Content Credentials</div>
                        <div className="sig-detail">
                          <span className="tag tag-neutral">No C2PA</span>
                        </div>
                      </div>
                    )}

                    {/* EXIF Check */}
                    {result.metadata?.has_exif ? (
                      <>
                        <div className="sig-value">
                          {result.metadata.camera_make || "Unknown"} {result.metadata.camera_model || ""}
                        </div>
                        <div className="sig-detail">
                          <span className="tag tag-blue">EXIF Data</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="sig-value text-tertiary">No Camera Data</div>
                        <div className="sig-detail">
                          <span className="tag tag-neutral">Metadata Stripped</span>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="sig">
                    <div className="sig-label">Visual Explanation</div>
                    {result.heatmap ? (
                      <>
                        <div className="sig-value">Grad-CAM Generated</div>
                        <div className="sig-detail">
                          <span className="tag tag-blue">Heatmap Available</span>
                        </div>
                      </>
                    ) : (
                      <div className="sig-value text-tertiary">Not Available</div>
                    )}
                  </div>

                  {result.text_consistency && (
                    <div className="sig full">
                      <div className="sig-label">Caption Consistency</div>
                      <div className="sig-value">
                        Score: {result.text_consistency.consistency_score?.toFixed(4) ?? "N/A"}
                      </div>
                      <div className="sig-detail" style={{ marginTop: 8 }}>
                        <span className={`tag ${result.text_consistency.likely_consistent ? "tag-green" : "tag-amber"}`}>
                          {result.text_consistency.likely_consistent ? "Semantically Consistent" : "Potential Mismatch"}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="disclaimer">
                  <strong>Note:</strong> This tool uses a probabilistic model and is not definitive proof. Results should be evaluated alongside other evidence.
                </div>

                <div className="card-actions">
                  <button className="btn-full" onClick={reset}>
                    Analyze Another Image
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="error-msg">
            {error}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          SignalScope — Developed for <a href="https://www.sih.gov.in/" target="_blank" rel="noreferrer">SIH 2026</a>.
          View the <a href="https://github.com/taher51-lang/SignalScope" target="_blank" rel="noreferrer">source code</a>.
        </p>
      </footer>
    </>
  );
}

export default App;