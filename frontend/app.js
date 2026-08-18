const state = {
  runs: [],
  selectedRun: null,
  charts: {
    training: null,
    valLoss: null,
    valF1: null,
  },
};

const palette = {
  lineA: "#0f8b8d",
  lineB: "#f28f3b",
  lineC: "#1f5f99",
};

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toFixed(digits);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function updateHealthBadge(health) {
  const el = document.getElementById("health");
  if (!health.model_loaded) {
    el.textContent = `Backend OK | Model not loaded`;
    el.style.background = "#fff4e8";
    return;
  }
  el.textContent = `Backend OK | Model ready`;
  el.style.background = "#e9f9f4";
}

function renderRunsTable() {
  const tbody = document.querySelector("#runs-table tbody");
  tbody.innerHTML = "";

  state.runs.forEach((run) => {
    const tr = document.createElement("tr");
    if (run.id === state.selectedRun) tr.classList.add("selected");
    tr.innerHTML = `
      <td>${run.name}</td>
      <td>${fmt(run.validation_f1)}</td>
      <td>${fmt(run.validation_loss)}</td>
      <td>${fmt(run.test_f1)}</td>
      <td>${fmt(run.test_loss)}</td>
    `;
    tr.addEventListener("click", async () => {
      state.selectedRun = run.id;
      renderRunsTable();
      await loadSelectedRun(run.id);
    });
    tbody.appendChild(tr);
  });
}

function renderCards(metrics) {
  const val = metrics.validation || {};
  const test = metrics.test || {};

  document.getElementById("val-f1").textContent = fmt(val.eval_f1);
  document.getElementById("val-loss").textContent = fmt(val.eval_loss);
  document.getElementById("test-f1").textContent = fmt(test.eval_f1);
  document.getElementById("test-loss").textContent = fmt(test.eval_loss);
}

function createLineChart(canvasId, label, points, color) {
  const element = document.getElementById(canvasId);
  return new Chart(element, {
    type: "line",
    data: {
      datasets: [
        {
          label,
          data: points,
          borderColor: color,
          backgroundColor: color,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      resizeDelay: 150,
      animation: {
        duration: 220,
      },
      parsing: false,
      scales: {
        x: { type: "linear" },
      },
      plugins: {
        legend: { display: false },
      },
      maintainAspectRatio: false,
    },
  });
}

function destroyCharts() {
  Object.keys(state.charts).forEach((key) => {
    if (state.charts[key]) {
      state.charts[key].destroy();
      state.charts[key] = null;
    }
  });
}

function renderCharts(history) {
  destroyCharts();
  state.charts.training = createLineChart(
    "training-chart",
    "Training Loss",
    history.training_loss || [],
    palette.lineA
  );
  state.charts.valLoss = createLineChart(
    "val-loss-chart",
    "Validation Loss",
    history.validation_loss || [],
    palette.lineB
  );
  state.charts.valF1 = createLineChart(
    "val-f1-chart",
    "Validation F1",
    history.validation_f1 || [],
    palette.lineC
  );
}

function renderConfusionImage(runId, artifacts) {
  const image = document.getElementById("confusion-image");
  const empty = document.getElementById("confusion-empty");
  const preferredByRun = {
    hyperparameter_tuning: "hp_experiment_1.png",
  };

  const preferred = preferredByRun[runId];
  const candidate = (preferred && artifacts.includes(preferred) ? preferred : null)
    || artifacts.find((name) => name.toLowerCase().includes("confusion") && name.endsWith(".png"))
    || artifacts.find((name) => name.toLowerCase().endsWith(".png"));
  if (!candidate) {
    image.style.display = "none";
    empty.style.display = "block";
    return;
  }
  image.src = `/api/runs/${runId}/artifacts/${candidate}`;
  image.style.display = "block";
  empty.style.display = "none";
}

async function loadSelectedRun(runId) {
  const [metrics, history, artifacts] = await Promise.all([
    fetchJson(`/api/runs/${runId}/metrics`),
    fetchJson(`/api/runs/${runId}/history`),
    fetchJson(`/api/runs/${runId}/artifacts`),
  ]);
  renderCards(metrics);
  renderCharts(history);
  renderConfusionImage(runId, artifacts.artifacts || []);
}

async function loadRuns() {
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs || [];
  state.selectedRun = state.runs.length > 0 ? state.runs[0].id : null;
  renderRunsTable();
  if (state.selectedRun) {
    await loadSelectedRun(state.selectedRun);
  }
}

async function initPredictForm() {
  const form = document.getElementById("predict-form");
  const errorEl = document.getElementById("predict-error");
  const outputEl = document.getElementById("compressed-text");
  const statsEl = document.getElementById("predict-stats");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorEl.textContent = "";
    outputEl.textContent = "Loading...";
    statsEl.textContent = "";

    const question = document.getElementById("question").value;
    const context = document.getElementById("context").value;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, context }),
      });

      if (!response.ok) {
        const details = await response.text();
        throw new Error(details || "Prediction failed");
      }

      const result = await response.json();
      outputEl.textContent = result.compressed_text || "";
      const stats = result.stats || {};
      statsEl.textContent = `Tokens in: ${stats.input_tokens ?? "-"} | Tokens kept: ${stats.kept_tokens ?? "-"} | Compression: ${fmt((stats.compression_ratio ?? 0) * 100, 1)}%`;
    } catch (err) {
      outputEl.textContent = "-";
      errorEl.textContent = err.message;
    }
  });
}

async function bootstrap() {
  try {
    const health = await fetchJson("/api/health");
    updateHealthBadge(health);
    await loadRuns();
    await initPredictForm();
  } catch (error) {
    const healthEl = document.getElementById("health");
    healthEl.textContent = "Backend unavailable";
    healthEl.style.background = "#ffe9e9";
    console.error(error);
  }
}

function startApp() {
  bootstrap().catch((error) => {
    const healthEl = document.getElementById("health");
    healthEl.textContent = "Backend unavailable";
    healthEl.style.background = "#ffe9e9";
    console.error(error);
  });
}

if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(startApp);
} else if (document.readyState === "complete") {
  startApp();
} else {
  window.addEventListener("load", startApp, { once: true });
}
