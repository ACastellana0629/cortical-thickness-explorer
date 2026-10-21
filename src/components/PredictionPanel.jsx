import { useEffect, useMemo, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const GROUP_LABELS = {
  CN: "Cognitively Normal",
  MCI: "Mild Cognitive Impairment",
  AD_TRAJECTORY: "AD Trajectory",
};

const GROUP_COLORS = {
  CN: "#59d8ff",
  MCI: "#ffd166",
  AD_TRAJECTORY: "#ff5ca8",
};

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function PredictionPanel() {
  const [profiles, setProfiles] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadModelData() {
      try {
        const [profilesResponse, metricsResponse] = await Promise.all([
          fetch("/data/demo_profiles.json"),
          fetch("/data/model_metrics.json"),
        ]);

        if (!profilesResponse.ok || !metricsResponse.ok) {
          throw new Error("Unable to load the model demonstration data.");
        }

        const profilesData = await profilesResponse.json();
        const metricsData = await metricsResponse.json();

        setProfiles(profilesData.profiles);
        setMetrics(metricsData);
        setSelectedProfileId(profilesData.profiles[0]?.id || "");
      } catch (loadError) {
        setError(loadError.message);
      } finally {
        setLoading(false);
      }
    }

    loadModelData();
  }, []);

  const selectedProfile = useMemo(
    () =>
      profiles.find(
        (profile) => profile.id === selectedProfileId,
      ),
    [profiles, selectedProfileId],
  );

  async function requestPrediction() {
    if (!selectedProfile) {
      return;
    }

    setPredicting(true);
    setPrediction(null);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          features: selectedProfile.features,
        }),
      });

      if (!response.ok) {
        const failure = await response.json();
        throw new Error(
          failure.detail?.message ||
            failure.detail ||
            "The prediction request failed.",
        );
      }

      const result = await response.json();
      setPrediction(result);
    } catch (predictionError) {
      setError(
        `${predictionError.message} Make sure the FastAPI backend is running on port 8000.`,
      );
    } finally {
      setPredicting(false);
    }
  }

  if (loading) {
    return (
      <section className="panel model-lab">
        <p>Loading neural-network system…</p>
      </section>
    );
  }

  if (!metrics) {
    return (
      <section className="panel model-lab">
        <p className="prediction-error">{error}</p>
      </section>
    );
  }

  return (
    <section className="panel model-lab">
      <div className="panel-heading model-lab-heading">
        <div>
          <p className="section-label">Neural prediction system</p>
          <h2>Cognitive Trajectory Classifier</h2>
        </div>

        <div className="model-status">
          <span className="status-light" />
          PyTorch MLP ready
        </div>
      </div>

      <p className="model-introduction">
        Select an aggregate demonstration profile and send its 68
        cortical-thickness measurements to the trained neural network.
        These profiles are group-level prototypes, not participant records.
      </p>

      <div className="prediction-grid">
        <div className="prediction-console">
          <p className="console-label">01 / Select input profile</p>

          <label className="profile-selector">
            Aggregate cortical pattern
            <select
              value={selectedProfileId}
              onChange={(event) => {
                setSelectedProfileId(event.target.value);
                setPrediction(null);
                setError("");
              }}
            >
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.display_name}
                </option>
              ))}
            </select>
          </label>

          {selectedProfile && (
            <div className="profile-summary">
              <span>Input dimensions</span>
              <strong>
                {Object.keys(selectedProfile.features).length} regions
              </strong>
              <small>{selectedProfile.source}</small>
            </div>
          )}

          <button
            className="predict-button"
            type="button"
            onClick={requestPrediction}
            disabled={!selectedProfile || predicting}
          >
            {predicting
              ? "Running neural network…"
              : "Generate prediction"}
          </button>

          {error && <p className="prediction-error">{error}</p>}

          <p className="research-warning">
            Research and educational prototype only. Predictions are not
            medical diagnoses.
          </p>
        </div>

        <div className="prediction-output">
          <p className="console-label">02 / Model response</p>

          {!prediction ? (
            <div className="prediction-placeholder">
              <span>Awaiting input</span>
              <p>
                Choose a profile and run the model to display its prediction.
              </p>
            </div>
          ) : (
            <>
              <div className="predicted-class">
                <span>Predicted trajectory</span>
                <strong>
                  {GROUP_LABELS[prediction.predicted_class]}
                </strong>
              </div>

              <div className="probability-list">
                {Object.entries(prediction.probabilities).map(
                  ([group, probability]) => (
                    <div className="probability-row" key={group}>
                      <div className="probability-heading">
                        <span>{GROUP_LABELS[group]}</span>
                        <strong>{formatPercent(probability)}</strong>
                      </div>

                      <div className="probability-track">
                        <div
                          className="probability-fill"
                          style={{
                            width: formatPercent(probability),
                            backgroundColor: GROUP_COLORS[group],
                          }}
                        />
                      </div>
                    </div>
                  ),
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="training-monitor">
        <div className="monitor-heading">
          <div>
            <p className="console-label">03 / Training monitor</p>
            <h3>Saved Model Performance</h3>
          </div>

          <span>Best epoch: {metrics.mlp.best_epoch}</span>
        </div>

        <div className="metric-cards">
          <div>
            <span>MLP test accuracy</span>
            <strong>
              {formatPercent(metrics.mlp.test.accuracy)}
            </strong>
          </div>

          <div>
            <span>Logistic benchmark</span>
            <strong>
              {formatPercent(
                metrics.logistic_regression_benchmark.accuracy,
              )}
            </strong>
          </div>

          <div>
            <span>Validation accuracy</span>
            <strong>
              {formatPercent(metrics.mlp.validation.accuracy)}
            </strong>
          </div>

          <div>
            <span>Test macro F1</span>
            <strong>
              {metrics.mlp.test.macro_f1.toFixed(3)}
            </strong>
          </div>
        </div>

        <img
          className="training-curves"
          src="/data/training_curves.png"
          alt="Training and validation loss and accuracy across epochs"
        />

        <p className="model-limitation">
          The model distinguishes the CN and AD Trajectory patterns more
          reliably than MCI. MCI remains difficult because it is
          heterogeneous and overlaps with both normal aging and dementia.
        </p>
      </div>
    </section>
  );
}

export default PredictionPanel;