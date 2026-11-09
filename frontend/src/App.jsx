import { useState } from "react"
import "./App.css"

const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "")

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [predictions, setPredictions] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState("")

  async function HandlePredict() {
    setErrorMessage("")
    setPredictions([])

    if (selectedFile === null) {
      setErrorMessage("Please select an engine readings file.")
      return
    }

    setIsLoading(true)

    const formData = new FormData()
    formData.append("file", selectedFile)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        setErrorMessage(
          typeof data.detail === "string"
            ? data.detail
            : "The backend could not process this file."
        )
        return
      }

      setPredictions(data.predictions)
    } catch (error) {
      console.error(error)
      setErrorMessage(
        "Could not complete the request. Check your connection and try again."
      )
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#predict">
          <span className="brand-icon" aria-hidden="true">
            ✈
          </span>
          Turbofan RUL Predictor
        </a>

        <a className="nav-link" href="#predict">
          Predict
        </a>
      </header>

      <main id="predict">
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-copy">
            <p className="eyebrow">AIRCRAFT ENGINE ANALYTICS</p>

            <h1 id="page-title">
              Turbofan
              <br />
              RUL Predictor
            </h1>

            <p className="hero-description">
              Estimate Remaining Useful Life (RUL)—the number of cycles an
              engine may operate before failure—from its sensor readings.
            </p>
          </div>
        </section>

        <div className="page-content">
          <section className="upload-panel" aria-labelledby="upload-title">
            <div className="section-heading">
              <span className="step-number" aria-hidden="true">
                01
              </span>

              <div>
                <h2 id="upload-title">Upload engine readings</h2>
                <p>Use a NASA C-MAPSS FD001-format text file.</p>
              </div>
            </div>

            <div className="upload-controls">
              <div className="file-field">
                <label htmlFor="engine-file">Engine readings file</label>

                <input
                  id="engine-file"
                  type="file"
                  accept=".txt"
                  disabled={isLoading}
                  aria-describedby="file-help"
                  onChange={(event) => {
                    setSelectedFile(event.target.files[0] ?? null)
                    setErrorMessage("")
                    setPredictions([])
                  }}
                />
              </div>

              <button
                type="button"
                onClick={HandlePredict}
                disabled={isLoading}
              >
                {isLoading ? "Predicting…" : "Predict RUL"}
              </button>
            </div>

            <p id="file-help" className="file-help">
              Plain-text .txt · 26 numeric columns · No header row · Maximum
              10 MB. Include consecutive cycles for each engine.
            </p>

            <p className="error-message" role="alert">
              {errorMessage}
            </p>

            <p className="loading-message" role="status">
              {isLoading
                ? "Processing engine readings. Please wait…"
                : ""}
            </p>
          </section>

          <section
            className="results-panel"
            aria-labelledby="results-title"
            aria-busy={isLoading}
          >
            <div className="results-heading">
              <div>
                <h2 id="results-title">Prediction results</h2>
                <p>
                  Estimated remaining cycles after each engine’s last
                  recorded cycle.
                </p>
              </div>

              {predictions.length > 0 && (
                <span className="engine-count">
                  {predictions.length} engines
                </span>
              )}
            </div>

            {predictions.length > 0 ? (
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Engine ID</th>
                      <th scope="col">Last Recorded Cycle</th>
                      <th scope="col">Predicted RUL (cycles)</th>
                    </tr>
                  </thead>

                  <tbody>
                    {predictions.map((prediction) => (
                      <tr key={prediction.Engine_ID}>
                        <td>{prediction.Engine_ID}</td>
                        <td>{prediction.Cycle}</td>
                        <td className="rul-value">
                          {prediction.Predicted_RUL.toFixed(1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                <span className="empty-icon" aria-hidden="true">
                  ↗
                </span>

                <h3>
                  {isLoading
                    ? "Calculating predictions"
                    : "Your predictions will appear here"}
                </h3>

                <p>
                  {isLoading
                    ? "Results will appear once processing finishes."
                    : "Choose an engine readings file, then click Predict RUL."}
                </p>
              </div>
            )}
          </section>

          <footer className="page-footer">
            NASA C-MAPSS FD001 · Random Forest model
            <span>
              Predictions are estimates for this dataset, not maintenance
              instructions.
            </span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App