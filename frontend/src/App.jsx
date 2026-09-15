import { useState, useCallback } from "react";

const API_URL = "http://localhost:8000/predict";

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
      setError("Please upload a valid image file.");
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

  const handleSubmit = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", image);

    const endpoint = caption.trim()
      ? "http://localhost:8000/predict_multimodal"
      : "http://localhost:8000/predict";

    if (caption.trim()) formData.append("caption", caption.trim());

    try {
      const res = await fetch(endpoint, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Prediction failed. Please try again.");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  const reset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setShowHeatmap(false);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center px-4 py-12">
      <div className="w-full max-w-xl">
        <h1 className="text-3xl font-bold text-gray-900 text-center">SignalScope</h1>
        <p className="text-gray-500 text-center mt-2 mb-10">
          Telling real from synthetic — upload an image to check its likely origin.
        </p>

        {!preview && (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-2xl p-12 text-center transition-colors cursor-pointer
              ${dragActive ? "border-indigo-500 bg-indigo-50" : "border-gray-300 bg-white"}`}
            onClick={() => document.getElementById("fileInput").click()}
          >
            <p className="text-gray-600 font-medium">Drag & drop an image here</p>
            <p className="text-gray-400 text-sm mt-1">or click to browse</p>
            <input
              id="fileInput"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>
        )}

        {preview && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            {/* Image display with heatmap toggle */}
            <div className="relative">
              <img
                src={showHeatmap && result?.heatmap ? result.heatmap : preview}
                alt="preview"
                className="w-full max-h-80 object-contain rounded-lg mb-3"
              />
              {result?.heatmap && (
                <div className="flex justify-center gap-2 mb-4">
                  <button
                    onClick={() => setShowHeatmap(false)}
                    className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${!showHeatmap
                      ? "bg-gray-900 text-white border-gray-900"
                      : "bg-white text-gray-600 border-gray-300"
                      }`}
                  >
                    Original
                  </button>
                  <button
                    onClick={() => setShowHeatmap(true)}
                    className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${showHeatmap
                      ? "bg-gray-900 text-white border-gray-900"
                      : "bg-white text-gray-600 border-gray-300"
                      }`}
                  >
                    Explanation Heatmap
                  </button>
                </div>
              )}
            </div>

            {!result && (
              <div>
                <input
                  type="text"
                  placeholder="Optional: add a caption to check image-text consistency"
                  value={caption}
                  onChange={(e) => setCaption(e.target.value)}
                  className="w-full border border-gray-300 rounded-xl px-4 py-2 text-sm mb-3 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <div className="flex gap-3">
                  <button
                    onClick={handleSubmit}
                    disabled={loading}
                    className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-medium py-3 rounded-xl transition-colors"
                  >
                    {loading ? "Analyzing..." : "Check Authenticity"}
                  </button>
                  <button
                    onClick={reset}
                    className="px-4 py-3 rounded-xl border border-gray-300 text-gray-600 hover:bg-gray-50"
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}
            {result && (
              <div className="space-y-4">
                <div
                  className={`rounded-xl p-5 border ${result.label === "AI-GENERATED"
                    ? "bg-amber-50 border-amber-200"
                    : "bg-green-50 border-green-200"
                    }`}
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-sm font-semibold px-3 py-1 rounded-full ${result.label === "AI-GENERATED"
                        ? "bg-amber-100 text-amber-800"
                        : "bg-green-100 text-green-800"
                        }`}
                    >
                      {result.label === "AI-GENERATED" ? "Likely AI-generated" : "Likely Real"}
                    </span>
                    <span className="text-gray-500 text-sm">
                      {(result.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>

                  <div className="mt-4 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${result.label === "AI-GENERATED" ? "bg-amber-500" : "bg-green-500"
                        }`}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                  {result.metadata && (
                    <div className="text-xs text-gray-500 border-t border-gray-200 pt-3 mt-3">
                      <p className="font-medium text-gray-700 mb-1">Metadata check</p>
                      <p>{result.metadata.has_exif ? `Camera: ${result.metadata.camera_make || "Unknown"} ${result.metadata.camera_model || ""}` : "No EXIF metadata found."}</p>
                    </div>
                  )}

                  {result.text_consistency && (
                    <div className="text-xs text-gray-500 border-t border-gray-200 pt-3 mt-3">
                      <p className="font-medium text-gray-700 mb-1">Caption consistency</p>
                      <p>Score: {result.text_consistency.consistency_score} — {result.text_consistency.likely_consistent ? "Likely consistent" : "Possibly mismatched"}</p>
                    </div>
                  )}

                  <p className="text-xs text-gray-500 mt-3">
                    This is a probabilistic assessment, not a definitive claim. Toggle above to see which regions influenced this verdict.
                  </p>
                </div>

                <button
                  onClick={reset}
                  className="w-full py-3 rounded-xl border border-gray-300 text-gray-600 hover:bg-gray-50"
                >
                  Try another image
                </button>
              </div>
            )}
          </div>
        )}

        {error && (
          <p className="text-red-500 text-sm text-center mt-4">{error}</p>
        )}
      </div>
    </div>
  );
}

export default App;