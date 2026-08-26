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

async function initPredictForm() {
  const form = document.getElementById("predict-form");
  const errorEl = document.getElementById("predict-error");
  const outputEl = document.getElementById("compressed-text");
  const statsEl = document.getElementById("predict-stats");
  const answerCard = document.getElementById("answer-card");
  const answerText = document.getElementById("answer-text");
  const answerSourceCount = document.getElementById("answer-source-count");
  const copyAnswerBtn = document.getElementById("copy-answer-btn");
  const sourcesSection = document.getElementById("sources-section");
  const sourcesList = document.getElementById("sources-list");
  const promptDetails = document.getElementById("prompt-details");
  const promptText = document.getElementById("prompt-text");
  const retrieveRagCheckbox = document.getElementById("retrieve-rag");
  const contextLabel = document.getElementById("context-label");
  const contextTextarea = document.getElementById("context");
  const questionTextarea = document.getElementById("question");
  const generateAnswerCheckbox = document.getElementById("generate-answer");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");

  // Toggle context requirement when RAG mode is enabled/disabled
  retrieveRagCheckbox.addEventListener("change", () => {
    if (retrieveRagCheckbox.checked) {
      contextLabel.style.opacity = "0.6";
      contextTextarea.placeholder = "RAG mode active: Knowledge base will be queried automatically.";
      btnText.textContent = "🔍 Search RAG & Answer";
    } else {
      contextLabel.style.opacity = "1";
      contextTextarea.placeholder = "संदर्भ अनुच्छेद यहाँ चिपकाएँ...";
      btnText.textContent = generateAnswerCheckbox.checked
        ? "🚀 Compress & Generate Answer"
        : "⚡ Compress Context";
    }
  });

  generateAnswerCheckbox.addEventListener("change", () => {
    if (!retrieveRagCheckbox.checked) {
      btnText.textContent = generateAnswerCheckbox.checked
        ? "🚀 Compress & Generate Answer"
        : "⚡ Compress Context";
    }
  });

  // Preset quick buttons
  document.querySelectorAll(".chip-btn").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q");
      const ctx = chip.getAttribute("data-ctx");
      const rag = chip.getAttribute("data-rag") === "true";

      if (q) questionTextarea.value = q;
      if (rag) {
        retrieveRagCheckbox.checked = true;
        retrieveRagCheckbox.dispatchEvent(new Event("change"));
      } else {
        retrieveRagCheckbox.checked = false;
        retrieveRagCheckbox.dispatchEvent(new Event("change"));
        if (ctx) contextTextarea.value = ctx;
      }
    });
  });

  // Copy Answer button
  if (copyAnswerBtn) {
    copyAnswerBtn.addEventListener("click", async () => {
      if (!answerText.textContent) return;
      try {
        await navigator.clipboard.writeText(answerText.textContent);
        const orig = copyAnswerBtn.textContent;
        copyAnswerBtn.textContent = "✅ Copied!";
        setTimeout(() => {
          copyAnswerBtn.textContent = orig;
        }, 1800);
      } catch (e) {
        console.error("Clipboard copy failed:", e);
      }
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorEl.textContent = "";
    errorEl.style.display = "none";
    outputEl.textContent = "Processing...";
    statsEl.innerHTML = "";
    answerCard.style.display = "none";
    sourcesSection.style.display = "none";
    sourcesList.innerHTML = "";
    promptDetails.style.display = "none";

    submitBtn.disabled = true;
    btnSpinner.style.display = "inline-block";

    const question = questionTextarea.value.trim();
    const context = contextTextarea.value.trim();
    const generate_answer = generateAnswerCheckbox.checked;
    const retrieve_rag = retrieveRagCheckbox.checked;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          context: retrieve_rag ? null : context,
          generate_answer,
          retrieve_rag,
          top_k: 3,
        }),
      });

      if (!response.ok) {
        const details = await response.text();
        throw new Error(details || `Prediction failed (${response.status})`);
      }

      const result = await response.json();

      // Render compressed text
      outputEl.textContent = result.compressed_text || "-";

      // Render stats badges
      const stats = result.stats || {};
      const ratioPct = fmt((stats.compression_ratio ?? 0) * 100, 1);
      statsEl.innerHTML = `
        <span class="stat-pill">Tokens In: <strong>${stats.input_tokens ?? "-"}</strong></span>
        <span class="stat-pill">Tokens Kept: <strong>${stats.kept_tokens ?? "-"}</strong></span>
        <span class="stat-pill highlight">Compressed: <strong>${ratioPct}%</strong></span>
      `;

      // Render Answer Card if answer exists
      if (result.answer) {
        answerText.textContent = result.answer;
        answerCard.style.display = "block";
        if (result.sources && result.sources.length > 0) {
          answerSourceCount.textContent = `(${result.sources.length} sources used)`;
        } else {
          answerSourceCount.textContent = "";
        }
      }

      // Render RAG Sources if present
      if (result.sources && result.sources.length > 0) {
        sourcesSection.style.display = "block";
        sourcesList.innerHTML = "";
        result.sources.forEach((src, idx) => {
          const card = document.createElement("div");
          card.className = "source-card";
          card.innerHTML = `
            <strong>Source ${idx + 1}: ${src.title || "Untitled"}</strong>
            <span class="hint">Score: ${src.score ?? "-"}</span>
            ${src.source_url ? `<div><a href="${src.source_url}" target="_blank" rel="noopener">${src.source_url}</a></div>` : ""}
          `;
          sourcesList.appendChild(card);
        });
      }

      // Render prompt details
      if (result.prompt) {
        promptText.textContent = result.prompt;
        promptDetails.style.display = "block";
      }

    } catch (err) {
      outputEl.textContent = "-";
      errorEl.textContent = err.message;
      errorEl.style.display = "block";
    } finally {
      submitBtn.disabled = false;
      btnSpinner.style.display = "none";
    }
  });
}

async function bootstrap() {
  try {
    const health = await fetchJson("/api/health");
    updateHealthBadge(health);
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

