const view = document.querySelector("#view");
const toast = document.querySelector("#toast");

const genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Animation", "Thriller"];

const state = {
  activeView: "for-you",
  userId: 42,
  health: null,
  recommendations: null,
  coldRecommendations: null,
  selectedMovieId: null,
  selectedGenres: new Set(),
  likedMovies: new Map(),
  searchResults: [],
  searchTimer: null,
  clickedMovies: new Set(),
  sessionId: window.sessionStorage.getItem("recsys-session") || crypto.randomUUID(),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compactNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return new Intl.NumberFormat("en", {
    notation: number >= 10000 ? "compact" : "standard",
    maximumFractionDigits: 1,
  }).format(number);
}

function preciseNumber(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function variantLabel(value) {
  if (value === "wide_and_deep") return "Wide & Deep";
  if (value === "dcn") return "DCN";
  if (!value) return "Fallback";
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function inferenceName() {
  const inference = state.health?.inference;
  if (!inference) return "Unavailable";
  return inference.backend ? String(inference.backend).toUpperCase() : "Local";
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("visible"), 2800);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP fallback when a response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

function setLoading(message) {
  view.innerHTML = `
    <section class="loading-state">
      <span class="loader" aria-hidden="true"></span>
      <p>${escapeHtml(message)}</p>
    </section>`;
}

function setError(error, retryView = state.activeView) {
  view.innerHTML = `
    <section class="error-state">
      <span class="eyebrow">REQUEST FAILED</span>
      <h1>Unable to load this view</h1>
      <p>${escapeHtml(error.message)}</p>
      <button class="secondary-button" type="button" data-retry="${retryView}">Retry request</button>
    </section>`;
}

function systemRibbon() {
  const health = state.health || {};
  const cacheHealthy = Boolean(health.cache_healthy);
  const inferenceReady = Boolean(health.inference?.ready);
  return `
    <section class="system-ribbon" aria-label="Serving system status">
      <div class="system-stat">
        <span>Known users</span>
        <strong>${compactNumber(health.known_users)}</strong>
      </div>
      <div class="system-stat">
        <span>Indexed items</span>
        <strong>${compactNumber(health.indexed_items)}</strong>
      </div>
      <div class="system-stat">
        <span>Feature cache</span>
        <strong><i class="status-dot ${cacheHealthy ? "" : "warn"}"></i>${cacheHealthy ? "Redis healthy" : "Memory fallback"}</strong>
      </div>
      <div class="system-stat">
        <span>Inference</span>
        <strong><i class="status-dot ${inferenceReady ? "" : "warn"}"></i>${escapeHtml(inferenceName())} · ${inferenceReady ? "ready" : "fallback"}</strong>
      </div>
    </section>`;
}

function normalizedScores(items) {
  const scores = items.map((item) => Number(item.score)).filter(Number.isFinite);
  if (!scores.length) return items.map(() => 0);
  const minimum = Math.min(...scores);
  const maximum = Math.max(...scores);
  if (minimum === maximum) return items.map(() => 100);
  return items.map((item) => {
    const score = Number(item.score);
    return Number.isFinite(score) ? ((score - minimum) / (maximum - minimum)) * 88 + 12 : 0;
  });
}

function resultsTable(items, impressionId) {
  if (!items.length) {
    return `
      <section class="empty-state">
        <span class="eyebrow">NO RESULTS</span>
        <p>The serving API returned an empty recommendation set.</p>
      </section>`;
  }

  const normalized = normalizedScores(items);
  return `
    <section class="table-shell" aria-label="Ranked recommendation results">
      <div class="results-table">
        <div class="result-header" role="row">
          <span>Rank</span>
          <span>Title</span>
          <span>Score</span>
          <span>Ranker</span>
          <span>Retrieval</span>
          <span>Reason</span>
        </div>
        ${items.map((movie, index) => {
          const movieId = Number(movie.movie_id);
          const selected = movieId === state.selectedMovieId ? " selected" : "";
          const clicked = state.clickedMovies.has(movieId) ? " clicked" : "";
          return `
            <button class="result-row${selected}${clicked}" type="button"
              data-movie-id="${movieId}"
              data-impression-id="${escapeHtml(impressionId || "")}"
              aria-pressed="${selected ? "true" : "false"}">
              <span class="rank-cell">${String(index + 1).padStart(2, "0")}</span>
              <span class="title-cell" title="${escapeHtml(movie.title || `Movie ${movieId}`)}">${escapeHtml(movie.title || `Movie ${movieId}`)}</span>
              <span class="score-cell">
                <span class="score-track"><i style="width:${normalized[index].toFixed(1)}%"></i></span>
                <code>${preciseNumber(movie.score, 3)}</code>
              </span>
              <span class="numeric-cell">${preciseNumber(movie.score)}</span>
              <span class="numeric-cell">${preciseNumber(movie.retrieval_score)}</span>
              <span class="reason-cell" title="${escapeHtml(movie.reason || "recommended")}">${escapeHtml(movie.reason || "recommended")}</span>
            </button>`;
        }).join("")}
      </div>
      <footer class="table-footer">
        <span>${items.length} rows · sorted by ranker score</span>
        <span>Click a row to inspect + record feedback</span>
      </footer>
    </section>`;
}

function selectedMovie(response) {
  const items = response?.recommendations || [];
  return items.find((movie) => Number(movie.movie_id) === state.selectedMovieId) || items[0] || null;
}

function requestTrace(response, resultCount) {
  const ranker = variantLabel(response?.variant);
  return `
    <section class="trace-card">
      <div class="card-head">
        <h2>Request trace</h2>
        <span>ONLINE PATH</span>
      </div>
      <div class="flow">
        <div class="flow-node">
          <span class="node-index">01</span>
          <strong>User ${escapeHtml(state.userId)}</strong>
          <code>subject</code>
        </div>
        <span class="connector">↓</span>
        <div class="flow-node">
          <span class="node-index">02</span>
          <strong>User tower</strong>
          <code>[1, 64]</code>
        </div>
        <span class="connector">↓</span>
        <div class="flow-node">
          <span class="node-index">03</span>
          <strong>FAISS retrieval</strong>
          <code>200 candidates</code>
        </div>
        <span class="connector">↓</span>
        <div class="flow-node">
          <span class="node-index">04</span>
          <strong>${escapeHtml(ranker)}</strong>
          <code>ranker</code>
        </div>
        <span class="connector">↓</span>
        <div class="flow-node">
          <span class="node-index">05</span>
          <strong>Top ${resultCount}</strong>
          <code>response</code>
        </div>
      </div>
    </section>`;
}

function servingContext(response) {
  return `
    <section class="context-card">
      <div class="card-head">
        <h2>Serving context</h2>
        <span>RESPONSE</span>
      </div>
      <dl class="context-list">
        <div><dt>cache</dt><dd>${response?.cache_hit ? "hit" : "miss"}</dd></div>
        <div><dt>experiment</dt><dd>ranking-model-v1</dd></div>
        <div><dt>variant</dt><dd>${escapeHtml(response?.variant || "fallback")}</dd></div>
        <div><dt>strategy</dt><dd>${escapeHtml(response?.strategy || "personalized")}</dd></div>
        <div><dt>model</dt><dd>${escapeHtml(response?.model_version || state.health?.model_version || "—")}</dd></div>
      </dl>
    </section>`;
}

function resultInspector(response) {
  const movie = selectedMovie(response);
  if (!movie) return "";
  return `
    <section class="inspector-card">
      <div class="card-head">
        <h2>Selected result</h2>
        <span>RANK INSPECTOR</span>
      </div>
      <div class="inspector-body">
        <h3>${escapeHtml(movie.title || `Movie ${movie.movie_id}`)}</h3>
        <code>movie_id: ${escapeHtml(movie.movie_id)}</code>
        <div class="inspector-scores">
          <div>
            <span>Ranker score</span>
            <strong>${preciseNumber(movie.score)}</strong>
          </div>
          <div>
            <span>Retrieval</span>
            <strong>${preciseNumber(movie.retrieval_score)}</strong>
          </div>
        </div>
      </div>
    </section>`;
}

function renderForYou() {
  const response = state.recommendations;
  const items = response?.recommendations || [];
  if (!items.length) {
    view.innerHTML = systemRibbon() + `
      <section class="empty-state"><span class="eyebrow">EMPTY RESPONSE</span><h1>No recommendations</h1><p>Try another positive MovieLens user ID.</p></section>`;
    return;
  }

  if (!state.selectedMovieId || !items.some((item) => Number(item.movie_id) === state.selectedMovieId)) {
    state.selectedMovieId = Number(items[0].movie_id);
  }

  view.innerHTML = `
    ${systemRibbon()}
    <section class="workspace">
      <section class="panel main-panel">
        <header class="page-heading">
          <div>
            <span class="eyebrow">ONLINE INFERENCE</span>
            <h1>Recommendations</h1>
            <p>Inspect retrieval, ranking, and serving context for a MovieLens user.</p>
          </div>
          <div class="status-tags">
            <span class="tag cyan">${escapeHtml(variantLabel(response.variant))}</span>
            <span class="tag ${response.cache_hit ? "green" : ""}">${response.cache_hit ? "CACHE HIT" : "FRESH RANKING"}</span>
          </div>
        </header>
        <form class="query-form" id="recommend-form">
          <div class="field-wrap">
            <label for="user-id">User ID</label>
            <input id="user-id" name="user-id" type="number" min="1" value="${state.userId}" required />
          </div>
          <button class="primary-button" type="submit">RUN INFERENCE →</button>
        </form>
        ${resultsTable(items, response.impression_id)}
      </section>
      <aside class="panel side-panel">
        <div class="side-stack">
          ${requestTrace(response, items.length)}
          ${servingContext(response)}
          ${resultInspector(response)}
        </div>
      </aside>
    </section>`;
}

async function loadForYou() {
  setLoading(`Running recommendation request for user ${state.userId}…`);
  try {
    const [health, recommendations] = await Promise.all([
      api("/health"),
      api(`/recommendations/${state.userId}?limit=10`),
    ]);
    state.health = health;
    state.recommendations = recommendations;
    state.selectedMovieId = null;
    state.clickedMovies.clear();
    renderForYou();
  } catch (error) {
    setError(error, "for-you");
  }
}

async function recordClick(movieId, impressionId) {
  if (!impressionId || state.clickedMovies.has(movieId)) return;
  try {
    const result = await api("/events/click", {
      method: "POST",
      body: JSON.stringify({ impression_id: impressionId, movie_id: movieId }),
    });
    state.clickedMovies.add(movieId);
    document.querySelectorAll(`[data-movie-id="${movieId}"]`).forEach((row) => row.classList.add("clicked"));
    showToast(`Feedback recorded · rank ${result.rank}`);
  } catch (error) {
    showToast(error.message);
  }
}

function profileSignals() {
  const selected = [...state.selectedGenres];
  if (!selected.length && !state.likedMovies.size) {
    return `<p class="helper-note">No profile signals selected. The service will use its popularity fallback.</p>`;
  }

  const genreRows = selected.map((genre) => `
    <div class="profile-line">
      <span>${escapeHtml(genre)}</span>
      <span class="profile-track"><i style="width:100%"></i></span>
      <code>ON</code>
    </div>`).join("");

  const likedRow = state.likedMovies.size ? `
    <div class="profile-line">
      <span>Liked titles</span>
      <span class="profile-track"><i style="width:100%"></i></span>
      <code>${state.likedMovies.size}</code>
    </div>` : "";

  return genreRows + likedRow;
}

function renderDiscover() {
  const response = state.coldRecommendations;
  const items = response?.recommendations || [];
  if (items.length && (!state.selectedMovieId || !items.some((item) => Number(item.movie_id) === state.selectedMovieId))) {
    state.selectedMovieId = Number(items[0].movie_id);
  }

  view.innerHTML = `
    ${systemRibbon()}
    <section class="cold-page">
      <header class="page-heading">
        <div>
          <span class="eyebrow">SESSION-BASED INFERENCE</span>
          <h1>Cold-start workbench</h1>
          <p>Construct a temporary taste profile from genres and catalog titles, then inspect the ranked response.</p>
        </div>
        <div class="status-tags">
          <span class="tag">CONTENT SIGNALS</span>
          <span class="tag">POPULARITY FALLBACK</span>
        </div>
      </header>

      <section class="cold-layout">
        <div class="selection-panel">
          <h2 class="section-title">01 · Define preference signals</h2>
          <p class="section-copy">Select any combination. All inputs remain scoped to this browser session.</p>
          <div class="genre-grid" aria-label="Genre preference signals">
            ${genres.map((genre) => `
              <button class="genre-button ${state.selectedGenres.has(genre) ? "selected" : ""}" type="button"
                data-genre="${genre}" aria-pressed="${state.selectedGenres.has(genre)}">
                + ${genre}
              </button>`).join("")}
          </div>

          <h2 class="section-title">02 · Add known positives</h2>
          <p class="section-copy">Search the indexed MovieLens catalog by title. This step is optional.</p>
          <div class="movie-search-wrap">
            <input class="movie-search" id="movie-search" type="search" autocomplete="off"
              placeholder="SEARCH CATALOG — E.G. TOY STORY" aria-label="Search catalog titles" />
            ${state.searchResults.length ? `
              <div class="search-results">
                ${state.searchResults.map((movie) => `
                  <button class="search-result" type="button" data-add-movie="${movie.movie_id}">
                    <strong>${escapeHtml(movie.title)}</strong>
                    <small>${escapeHtml((movie.genres || []).slice(0, 3).join(" · ") || "Movie")}</small>
                  </button>`).join("")}
              </div>` : ""}
          </div>
          <div class="liked-list">
            ${[...state.likedMovies.values()].map((movie) => `
              <article class="liked-card">
                <strong>${escapeHtml(movie.title)}</strong>
                <button type="button" data-remove-movie="${movie.movie_id}">REMOVE</button>
              </article>`).join("") || `<p class="helper-note">0 catalog titles added</p>`}
          </div>
        </div>

        <aside class="profile-panel">
          <h2 class="section-title">Profile signal</h2>
          <p class="section-copy">Active features sent to the cold-start endpoint.</p>
          <div class="profile-lines">${profileSignals()}</div>
          <button class="primary-button" id="generate-cold" type="button">GENERATE TOP 10 →</button>
          <p class="helper-note">POST /recommendations/cold-start</p>
        </aside>
      </section>

      ${items.length ? `
        <section class="cold-results">
          <header class="page-heading">
            <div>
              <span class="eyebrow">RESPONSE</span>
              <h2 class="section-title">Cold-start ranking</h2>
            </div>
            <div class="status-tags">
              <span class="tag cyan">${escapeHtml(response.strategy || "cold_start")}</span>
              <span class="tag ${response.cache_hit ? "green" : ""}">${response.cache_hit ? "CACHE HIT" : "FRESH RESPONSE"}</span>
            </div>
          </header>
          ${resultsTable(items, response.impression_id)}
        </section>` : ""}
    </section>`;
}

async function ensureHealth() {
  if (!state.health) state.health = await api("/health");
}

async function searchMovies(query) {
  if (query.trim().length < 2) {
    state.searchResults = [];
    renderDiscover();
    document.querySelector("#movie-search")?.focus();
    return;
  }
  try {
    const data = await api(`/catalog/search?q=${encodeURIComponent(query.trim())}&limit=6`);
    state.searchResults = data.results.filter((movie) => !state.likedMovies.has(Number(movie.movie_id)));
    renderDiscover();
    const input = document.querySelector("#movie-search");
    if (input) {
      input.value = query;
      input.focus();
      input.setSelectionRange(query.length, query.length);
    }
  } catch (error) {
    showToast(error.message);
  }
}

async function generateColdStart(button) {
  button.disabled = true;
  button.textContent = "RUNNING…";
  try {
    state.coldRecommendations = await api("/recommendations/cold-start", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        selected_genres: [...state.selectedGenres],
        liked_movie_ids: [...state.likedMovies.keys()],
        limit: 10,
      }),
    });
    window.sessionStorage.setItem("recsys-session", state.sessionId);
    state.selectedMovieId = null;
    state.clickedMovies.clear();
    renderDiscover();
    document.querySelector(".cold-results")?.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message);
    button.disabled = false;
    button.textContent = "GENERATE TOP 10 →";
  }
}

function variantMetrics(metrics, name) {
  return metrics?.variants?.[name] || {
    impressions: 0,
    clicks: 0,
    ctr: 0,
    discounted_clicks_per_impression: 0,
  };
}

function metricPanel(label, key, metrics, winner) {
  return `
    <article class="metric-panel ${winner ? "winner" : ""}">
      <div class="card-head">
        <h2>${escapeHtml(label)}</h2>
        <span>${escapeHtml(key)}</span>
      </div>
      <p>${winner ? "LEADING BY OBSERVED CTR" : "OBSERVED VARIANT"}</p>
      <div class="metric-value">${(Number(metrics.ctr) * 100).toFixed(2)}%</div>
      <dl class="metric-rows">
        <div><dt>impressions</dt><dd>${Number(metrics.impressions).toLocaleString()}</dd></div>
        <div><dt>clicks</dt><dd>${Number(metrics.clicks).toLocaleString()}</dd></div>
        <div><dt>discounted clicks / impression</dt><dd>${preciseNumber(metrics.discounted_clicks_per_impression)}</dd></div>
      </dl>
    </article>`;
}

async function loadExperiments() {
  setLoading("Reading recorded experiment events…");
  try {
    const [health, metrics] = await Promise.all([
      api("/health"),
      api("/experiments/ranking-model-v1/metrics"),
    ]);
    state.health = health;
    const wide = variantMetrics(metrics, "wide_and_deep");
    const dcn = variantMetrics(metrics, "dcn");
    const totalImpressions = Number(wide.impressions) + Number(dcn.impressions);
    const totalClicks = Number(wide.clicks) + Number(dcn.clicks);
    const hasTraffic = totalImpressions > 0;
    const wideWins = hasTraffic && Number(wide.ctr) > Number(dcn.ctr);
    const dcnWins = hasTraffic && Number(dcn.ctr) > Number(wide.ctr);
    const overallCtr = totalImpressions ? totalClicks / totalImpressions : 0;

    view.innerHTML = `
      ${systemRibbon()}
      <section class="metrics-page">
        <header class="page-heading">
          <div>
            <span class="eyebrow">EXPERIMENT · ranking-model-v1</span>
            <h1>A/B metrics</h1>
            <p>Observed interactions recorded by this demo. These values are not MovieLens historical metrics.</p>
          </div>
          <div class="status-tags">
            <span class="tag green">EVENTS RECORDING</span>
            <span class="tag">${hasTraffic ? "OBSERVED TRAFFIC" : "AWAITING CLICKS"}</span>
          </div>
        </header>

        <section class="metric-summary" aria-label="Experiment totals">
          <div class="summary-stat"><span>Total impressions</span><strong>${totalImpressions.toLocaleString()}</strong></div>
          <div class="summary-stat"><span>Total clicks</span><strong>${totalClicks.toLocaleString()}</strong></div>
          <div class="summary-stat"><span>Overall CTR</span><strong>${(overallCtr * 100).toFixed(2)}%</strong></div>
        </section>

        <section class="metric-grid">
          ${metricPanel("Wide & Deep", "wide_and_deep", wide, wideWins)}
          ${metricPanel("Deep & Cross Network", "dcn", dcn, dcnWins)}
        </section>
      </section>`;
  } catch (error) {
    setError(error, "experiments");
  }
}

function activateNav(name) {
  document.querySelectorAll(".main-nav [data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
}

async function switchView(name) {
  state.activeView = name;
  activateNav(name);
  if (name === "for-you") {
    await loadForYou();
  } else if (name === "discover") {
    setLoading("Loading catalog controls…");
    try {
      await ensureHealth();
      renderDiscover();
    } catch (error) {
      setError(error, "discover");
    }
  } else if (name === "experiments") {
    await loadExperiments();
  }
  view.focus({ preventScroll: true });
}

document.addEventListener("submit", (event) => {
  if (event.target.id !== "recommend-form") return;
  event.preventDefault();
  const value = Number(new FormData(event.target).get("user-id"));
  if (!Number.isInteger(value) || value <= 0) {
    showToast("Enter a positive MovieLens user ID.");
    return;
  }
  state.userId = value;
  switchView("for-you");
});

document.addEventListener("click", (event) => {
  const viewButton = event.target.closest("[data-view]");
  if (viewButton) {
    switchView(viewButton.dataset.view);
    return;
  }

  const retry = event.target.closest("[data-retry]");
  if (retry) {
    switchView(retry.dataset.retry);
    return;
  }

  const row = event.target.closest(".result-row[data-movie-id]");
  if (row) {
    const movieId = Number(row.dataset.movieId);
    state.selectedMovieId = movieId;
    if (state.activeView === "for-you") renderForYou();
    else renderDiscover();
    recordClick(movieId, row.dataset.impressionId);
    return;
  }

  const genre = event.target.closest("[data-genre]");
  if (genre) {
    const value = genre.dataset.genre;
    state.selectedGenres.has(value) ? state.selectedGenres.delete(value) : state.selectedGenres.add(value);
    state.coldRecommendations = null;
    state.selectedMovieId = null;
    renderDiscover();
    return;
  }

  const addMovie = event.target.closest("[data-add-movie]");
  if (addMovie) {
    const id = Number(addMovie.dataset.addMovie);
    const movie = state.searchResults.find((item) => Number(item.movie_id) === id);
    if (movie) state.likedMovies.set(id, movie);
    state.searchResults = [];
    state.coldRecommendations = null;
    renderDiscover();
    return;
  }

  const removeMovie = event.target.closest("[data-remove-movie]");
  if (removeMovie) {
    state.likedMovies.delete(Number(removeMovie.dataset.removeMovie));
    state.coldRecommendations = null;
    renderDiscover();
    return;
  }

  const generate = event.target.closest("#generate-cold");
  if (generate) generateColdStart(generate);
});

document.addEventListener("input", (event) => {
  if (event.target.id !== "movie-search") return;
  const query = event.target.value;
  window.clearTimeout(state.searchTimer);
  state.searchTimer = window.setTimeout(() => searchMovies(query), 220);
});

switchView("for-you");
