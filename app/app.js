const view = document.querySelector("#view");
const toast = document.querySelector("#toast");
const userForm = document.querySelector("#user-form");
const userInput = document.querySelector("#user-id");

const genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Animation", "Thriller"];
const genreIcons = {
  Action: "⚡",
  Comedy: "☺",
  Drama: "♡",
  "Sci-Fi": "◉",
  Animation: "▣",
  Thriller: "◎",
  Trending: "♨",
};

const state = {
  activeView: "for-you",
  userId: 42,
  health: null,
  recommendations: null,
  coldRecommendations: null,
  selectedGenres: new Set(),
  likedMovies: new Map(),
  searchResults: [],
  searchTimer: null,
  clickedMovies: new Set(),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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
      // Keep the HTTP fallback when the response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

function setLoading(message) {
  view.innerHTML = `
    <section class="loading-state">
      <div class="loading-orb" aria-hidden="true"></div>
      <p>${escapeHtml(message)}</p>
    </section>`;
}

function setError(error, retryView = state.activeView) {
  view.innerHTML = `
    <section class="error-state">
      <div>
        <h1>We couldn't load this view</h1>
        <p>${escapeHtml(error.message)}</p>
      </div>
      <button class="secondary-button" type="button" data-retry="${retryView}">Try again</button>
    </section>`;
}

function palette(movieId) {
  return Math.abs(Number(movieId) || 0) % 6;
}

function scoreLabel(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return "Ranked";
  return `score ${value.toFixed(3)}`;
}

function posterCard(movie, rank, impressionId) {
  const movieId = Number(movie.movie_id);
  const clicked = state.clickedMovies.has(movieId) ? " clicked" : "";
  return `
    <button class="poster-card palette-${palette(movieId)}${clicked}" type="button"
      data-movie-id="${movieId}" data-impression-id="${escapeHtml(impressionId || "")}">
      <span class="poster-art">
        <span class="poster-rank">#${rank}</span>
        <span class="poster-title-overlay">${escapeHtml(movie.title || `Movie ${movieId}`)}</span>
      </span>
      <span class="poster-details">
        <strong>${escapeHtml(movie.title || `Movie ${movieId}`)}</strong>
        <span>${escapeHtml(scoreLabel(movie.score))} · ${escapeHtml(movie.reason || "recommended")}</span>
      </span>
    </button>`;
}

function heroCard(movie, index, impressionId) {
  const movieId = Number(movie.movie_id);
  return `
    <button class="feature-card" type="button" data-palette="${index}"
      data-movie-id="${movieId}" data-impression-id="${escapeHtml(impressionId || "")}">
      <h2>${escapeHtml(movie.title || `Movie ${movieId}`)}</h2>
      <span class="feature-meta">
        <span class="play-label"><span class="play-icon">♥</span> Save this pick</span>
        <span class="feature-score">${escapeHtml(scoreLabel(movie.score))}</span>
      </span>
    </button>`;
}

function activateNav(name) {
  document.querySelectorAll("[data-view]").forEach((button) => {
    if (button.closest(".main-nav")) button.classList.toggle("active", button.dataset.view === name);
  });
  userForm.hidden = name !== "for-you";
}

async function recordClick(movieId, impressionId) {
  if (!impressionId || state.clickedMovies.has(movieId)) return;
  try {
    const result = await api("/events/click", {
      method: "POST",
      body: JSON.stringify({ impression_id: impressionId, movie_id: movieId }),
    });
    state.clickedMovies.add(movieId);
    document.querySelectorAll(`[data-movie-id="${movieId}"]`).forEach((card) => {
      card.classList.add("clicked");
    });
    showToast(`Saved your rank #${result.rank} pick.`);
  } catch (error) {
    showToast(error.message);
  }
}

function renderForYou() {
  const response = state.recommendations;
  const items = response?.recommendations || [];
  if (!items.length) {
    view.innerHTML = `<section class="empty-state"><h1>No recommendations yet</h1><p>Try another user ID or build a taste profile in Discover.</p></section>`;
    return;
  }
  const variant = response.variant === "wide_and_deep" ? "Wide & Deep" : response.variant?.toUpperCase();
  view.innerHTML = `
    <section class="feature-grid">
      ${items.slice(0, 2).map((movie, index) => heroCard(movie, index, response.impression_id)).join("")}
    </section>
    <section class="genre-strip" aria-label="Explore by genre">
      ${["Trending", ...genres.slice(0, 5)].map((genre) => `
        <button class="genre-pill" type="button" data-discover-genre="${genre === "Trending" ? "" : genre}">
          <span aria-hidden="true">${genreIcons[genre]}</span>${genre}
        </button>`).join("")}
    </section>
    <section class="section-heading">
      <div>
        <h2>Your ranked picks</h2>
        <p>Personalized for user ${state.userId}.</p>
      </div>
      <div class="status-chips">
        <span class="status-chip">${escapeHtml(variant || "Cold start")}</span>
        <span class="status-chip ${response.cache_hit ? "success" : ""}">${response.cache_hit ? "Cache hit" : "Fresh ranking"}</span>
        <span class="status-chip">${items.length} results</span>
      </div>
    </section>
    <section class="poster-grid" aria-label="Ranked movie recommendations">
      ${items.map((movie, index) => posterCard(movie, index + 1, response.impression_id)).join("")}
    </section>`;
}

async function loadForYou() {
  setLoading(`Ranking movies for user ${state.userId}…`);
  try {
    const [health, recommendations] = await Promise.all([
      api("/health"),
      api(`/recommendations/${state.userId}?limit=10`),
    ]);
    state.health = health;
    state.recommendations = recommendations;
    state.clickedMovies.clear();
    renderForYou();
  } catch (error) {
    setError(error, "for-you");
  }
}

function tasteRows() {
  const selected = [...state.selectedGenres];
  if (!selected.length) {
    return `<p class="helper-note">Select a few genres to shape your temporary profile.</p>`;
  }
  return selected.map((genre, index) => `
    <div class="taste-row">
      <strong>${escapeHtml(genre)}</strong>
      <span class="taste-track"><span style="width:${88 - index * 9}%"></span></span>
    </div>`).join("");
}

function renderDiscover() {
  const results = state.coldRecommendations?.recommendations || [];
  const impressionId = state.coldRecommendations?.impression_id;
  view.innerHTML = `
    <section class="page-intro">
      <h1>Tell us what you like</h1>
      <p>A few choices are enough to get started.</p>
    </section>
    <section class="preference-genres" aria-label="Movie genres">
      ${genres.map((genre) => `
        <button class="genre-pill ${state.selectedGenres.has(genre) ? "selected" : ""}" type="button" data-genre="${genre}">
          <span aria-hidden="true">${genreIcons[genre]}</span>${genre}
        </button>`).join("")}
    </section>
    <section class="preference-layout">
      <div class="glass-panel preference-panel">
        <h2>Add movies you already love</h2>
        <p class="disclaimer">Search the MovieLens catalog by title.</p>
        <div class="search-wrap">
          <input class="movie-search" id="movie-search" type="search" autocomplete="off" placeholder="Try ‘Toy Story’ or ‘Matrix’" />
          ${state.searchResults.length ? `<div class="search-results">
            ${state.searchResults.map((movie) => `
              <button class="search-result" type="button" data-add-movie="${movie.movie_id}">
                <strong>${escapeHtml(movie.title)}</strong>
                <small>${escapeHtml((movie.genres || []).slice(0, 2).join(" · ") || "Movie")}</small>
              </button>`).join("")}
          </div>` : ""}
        </div>
        <div class="liked-list">
          ${[...state.likedMovies.values()].map((movie) => `
            <article class="liked-card">
              <strong>${escapeHtml(movie.title)}</strong>
              <button type="button" data-remove-movie="${movie.movie_id}">Remove</button>
            </article>`).join("") || `<p class="helper-note">Movie selections are optional.</p>`}
        </div>
      </div>
      <aside class="glass-panel taste-panel">
        <h2>Your taste mix</h2>
        <div class="taste-bars">${tasteRows()}</div>
        <button class="primary-button" id="generate-cold" type="button">Create my recommendations →</button>
        <p class="helper-note">Content + popularity fallback</p>
      </aside>
    </section>
    ${results.length ? `
      <section class="cold-results">
        <div class="section-heading">
          <div><h2>Your first recommendations</h2><p>Built from this temporary taste profile.</p></div>
          <div class="status-chips">
            <span class="status-chip">Cold start</span>
            <span class="status-chip ${state.coldRecommendations.cache_hit ? "success" : ""}">${state.coldRecommendations.cache_hit ? "Cache hit" : "Fresh results"}</span>
          </div>
        </div>
        <div class="poster-grid">${results.map((movie, index) => posterCard(movie, index + 1, impressionId)).join("")}</div>
      </section>` : ""}`;
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
    state.searchResults = data.results.filter((movie) => !state.likedMovies.has(movie.movie_id));
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
  button.textContent = "Creating your picks…";
  try {
    state.coldRecommendations = await api("/recommendations/cold-start", {
      method: "POST",
      body: JSON.stringify({
        session_id: window.sessionStorage.getItem("recsys-session") || crypto.randomUUID(),
        selected_genres: [...state.selectedGenres],
        liked_movie_ids: [...state.likedMovies.keys()],
        limit: 10,
      }),
    });
    window.sessionStorage.setItem("recsys-session", state.coldRecommendations.subject.replace("session-", ""));
    state.clickedMovies.clear();
    renderDiscover();
    document.querySelector(".cold-results")?.scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    showToast(error.message);
    button.disabled = false;
    button.textContent = "Create my recommendations →";
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

function experimentCard(label, name, value, winner) {
  return `
    <article class="glass-panel experiment-card ${winner ? "winner" : ""}">
      <h2>${label}</h2>
      ${winner ? `<span class="leading-pill">Leading</span>` : ""}
      <div class="metric-grid">
        <div><span>Impressions</span><strong>${value.impressions.toLocaleString()}</strong></div>
        <div><span>Clicked sessions</span><strong>${value.clicks.toLocaleString()}</strong></div>
        <div><span>CTR</span><strong>${(value.ctr * 100).toFixed(2)}%</strong></div>
      </div>
    </article>`;
}

async function loadExperiments() {
  setLoading("Reading demo experiment events…");
  try {
    const [health, metrics] = await Promise.all([
      api("/health"),
      api("/experiments/ranking-model-v1/metrics"),
    ]);
    state.health = health;
    const wide = variantMetrics(metrics, "wide_and_deep");
    const dcn = variantMetrics(metrics, "dcn");
    const hasTraffic = wide.impressions + dcn.impressions > 0;
    const dcnWins = hasTraffic && dcn.ctr > wide.ctr;
    const wideWins = hasTraffic && wide.ctr > dcn.ctr;
    const maximum = Math.max(wide.ctr, dcn.ctr, 0.01);
    view.innerHTML = `
      <section class="page-intro">
        <h1>Which ranker connects better?</h1>
        <p>Live interactions from this demo.</p>
      </section>
      <section class="experiment-layout">
        ${experimentCard("Wide & Deep", "wide_and_deep", wide, wideWins)}
        ${experimentCard("DCN", "dcn", dcn, dcnWins)}
      </section>
      <section class="glass-panel summary-panel">
        <div>
          <h2>Experiment summary</h2>
          <div class="comparison-bars">
            <div class="comparison-row"><strong>DCN</strong><span class="comparison-track"><span style="width:${Math.max(3, dcn.ctr / maximum * 100)}%"></span></span><strong>${(dcn.ctr * 100).toFixed(2)}%</strong></div>
            <div class="comparison-row"><strong>Wide & Deep</strong><span class="comparison-track"><span style="width:${Math.max(3, wide.ctr / maximum * 100)}%"></span></span><strong>${(wide.ctr * 100).toFixed(2)}%</strong></div>
          </div>
          <p class="disclaimer">Demo interactions only — not MovieLens history.${hasTraffic ? "" : " Click a recommendation to begin collecting metrics."}</p>
        </div>
        <aside class="system-status">
          <span class="status-chip"><span class="status-dot"></span>${health.cache_healthy ? "Redis healthy" : "Memory fallback"}</span>
          <span class="status-chip"><span class="status-dot"></span>Events recording</span>
        </aside>
      </section>`;
  } catch (error) {
    setError(error, "experiments");
  }
}

async function switchView(name) {
  state.activeView = name;
  activateNav(name);
  if (name === "for-you") await loadForYou();
  else if (name === "discover") renderDiscover();
  else if (name === "experiments") await loadExperiments();
  view.focus({ preventScroll: true });
}

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
  const movieCard = event.target.closest("[data-movie-id]");
  if (movieCard) {
    recordClick(Number(movieCard.dataset.movieId), movieCard.dataset.impressionId);
    return;
  }
  const genre = event.target.closest("[data-genre]");
  if (genre) {
    const value = genre.dataset.genre;
    state.selectedGenres.has(value) ? state.selectedGenres.delete(value) : state.selectedGenres.add(value);
    state.coldRecommendations = null;
    renderDiscover();
    return;
  }
  const discoverGenre = event.target.closest("[data-discover-genre]");
  if (discoverGenre) {
    const value = discoverGenre.dataset.discoverGenre;
    if (value) state.selectedGenres.add(value);
    switchView("discover");
    return;
  }
  const addMovie = event.target.closest("[data-add-movie]");
  if (addMovie) {
    const id = Number(addMovie.dataset.addMovie);
    const movie = state.searchResults.find((item) => item.movie_id === id);
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

userForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = Number(userInput.value);
  if (!Number.isInteger(value) || value <= 0) {
    showToast("Enter a positive MovieLens user ID.");
    return;
  }
  state.userId = value;
  switchView("for-you");
});

switchView("for-you");
