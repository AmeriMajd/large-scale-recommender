(() => {
  const REPOSITORY_URL = "https://github.com/AmeriMajd/large-scale-recommender";
  const MODEL_VERSION =
    "cutoff-1466573687-seed-42-20261103T200303Z+" +
    "cutoff-1466573687-seed-42-20261103T200303Z-ranking-seed-42-20261104T223813Z";
  const METRICS_KEY = "recsys-portfolio-metrics-v1";

  const knownRecommendations = [
    { movie_id: 1580, title: "Men in Black (a.k.a. MIB) (1997)", score: 4.191734313964844, reason: "dcn", retrieval_score: 0.32449352741241455 },
    { movie_id: 2657, title: "Rocky Horror Picture Show, The (1975)", score: 2.821357011795044, reason: "dcn", retrieval_score: 0.36498045921325684 },
    { movie_id: 4758, title: "Soul Survivors (a.k.a. The Killer Cut) (2001)", score: 2.3530261516571045, reason: "dcn", retrieval_score: 0.7943758368492126 },
    { movie_id: 1680, title: "Sliding Doors (1998)", score: 2.339285135269165, reason: "dcn", retrieval_score: 0.33063796162605286 },
    { movie_id: 1586, title: "G.I. Jane (1997)", score: 1.7235440015792847, reason: "dcn", retrieval_score: 0.3134903311729431 },
    { movie_id: 1920, title: "Small Soldiers (1998)", score: 1.5469988584518433, reason: "dcn", retrieval_score: 0.43902087211608887 },
    { movie_id: 1735, title: "Great Expectations (1998)", score: 1.5200742483139038, reason: "dcn", retrieval_score: 0.31126976013183594 },
    { movie_id: 5400, title: "Sum of All Fears, The (2002)", score: 1.3521071672439575, reason: "dcn", retrieval_score: 0.3216125965118408 },
    { movie_id: 6367, title: "Down with Love (2003)", score: 1.3317418098449707, reason: "dcn", retrieval_score: 0.3923817574977875 },
    { movie_id: 4890, title: "Shallow Hal (2001)", score: 1.1448200941085815, reason: "dcn", retrieval_score: 0.3367554545402527 },
  ];

  const catalog = [
    { movie_id: 1, title: "Toy Story (1995)", genres: ["Adventure", "Animation", "Children", "Comedy", "Fantasy"] },
    { movie_id: 3114, title: "Toy Story 2 (1999)", genres: ["Adventure", "Animation", "Children", "Comedy", "Fantasy"] },
    { movie_id: 78499, title: "Toy Story 3 (2010)", genres: ["Adventure", "Animation", "Children", "Comedy", "Fantasy", "IMAX"] },
    { movie_id: 741, title: "Ghost in the Shell (Kôkaku kidôtai) (1995)", genres: ["Animation", "Sci-Fi"] },
    { movie_id: 32840, title: "Vincent (1982)", genres: ["Animation", "Fantasy"] },
    { movie_id: 81018, title: "Illusionist, The (L'illusionniste) (2010)", genres: ["Animation", "Drama"] },
    { movie_id: 2311, title: "2010: The Year We Make Contact (1984)", genres: ["Sci-Fi", "Thriller"] },
    { movie_id: 112623, title: "Dawn of the Planet of the Apes (2014)", genres: ["Action", "Drama", "Sci-Fi"] },
    { movie_id: 3354, title: "Mission to Mars (2000)", genres: ["Sci-Fi"] },
    { movie_id: 71057, title: "9 (2009)", genres: ["Adventure", "Animation", "Sci-Fi"] },
    { movie_id: 70533, title: "Evangelion: 1.0 You Are (Not) Alone (2007)", genres: ["Action", "Animation", "Sci-Fi"] },
    { movie_id: 136449, title: "Ghost in the Shell 2.0 (2008)", genres: ["Action", "Animation", "Sci-Fi"] },
    { movie_id: 71318, title: "Terra (a.k.a. Battle for Terra) (2007)", genres: ["Adventure", "Animation", "Sci-Fi"] },
    { movie_id: 2253, title: "Toys (1992)", genres: ["Comedy", "Fantasy"] },
    { movie_id: 5843, title: "Toy Soldiers (1991)", genres: ["Action", "Drama"] },
    { movie_id: 4929, title: "Toy, The (1982)", genres: ["Comedy"] },
  ];

  const coldStartScores = new Map([
    [741, 0.9443825131288572],
    [32840, 0.9411544927204682],
    [81018, 0.9361541203739147],
    [2311, 0.8992170043131723],
    [112623, 0.8986514742407087],
    [3354, 0.887570960861934],
    [71057, 0.7370483141462923],
    [70533, 0.7028180949912687],
    [136449, 0.6908610268522423],
    [71318, 0.6806491885143571],
  ]);

  const impressions = new Map();
  let active =
    window.location.protocol === "file:" ||
    window.location.hostname.endsWith(".pages.dev") ||
    new URLSearchParams(window.location.search).has("portfolio");

  function defaultMetrics() {
    return {
      wide_and_deep: { impressions: 64, clicks: 9, discounted_clicks: 6.12 },
      dcn: { impressions: 67, clicks: 11, discounted_clicks: 7.78 },
    };
  }

  function readMetrics() {
    try {
      const value = JSON.parse(window.localStorage.getItem(METRICS_KEY));
      if (value?.wide_and_deep && value?.dcn) return value;
    } catch {
      // A privacy setting may block local storage. The in-memory seed still works.
    }
    return defaultMetrics();
  }

  let metrics = readMetrics();

  function writeMetrics() {
    try {
      window.localStorage.setItem(METRICS_KEY, JSON.stringify(metrics));
    } catch {
      // Metrics remain available for this page session when storage is blocked.
    }
  }

  function publicMetrics() {
    const variants = {};
    for (const [name, value] of Object.entries(metrics)) {
      variants[name] = {
        impressions: value.impressions,
        clicks: value.clicks,
        ctr: value.impressions ? value.clicks / value.impressions : 0,
        discounted_clicks_per_impression: value.impressions
          ? value.discounted_clicks / value.impressions
          : 0,
      };
    }
    return { experiment_id: "ranking-model-v1", variants };
  }

  function syncChrome() {
    const banner = document.querySelector("#portfolio-banner");
    if (banner) banner.hidden = !active;
    const link = document.querySelector("#api-link");
    if (link && active) {
      link.href = REPOSITORY_URL;
      link.innerHTML = 'View source <span aria-hidden="true">↗</span>';
    }
    document.documentElement.dataset.portfolio = active ? "true" : "false";
  }

  function activate() {
    active = true;
    syncChrome();
  }

  function createImpression(variant, recommendations) {
    const impressionId = `portfolio-${variant}-${crypto.randomUUID()}`;
    impressions.set(impressionId, {
      variant,
      movieIds: recommendations.map((movie) => Number(movie.movie_id)),
      clicked: false,
    });
    if (metrics[variant]) {
      metrics[variant].impressions += 1;
      writeMetrics();
    }
    return impressionId;
  }

  function knownUserResponse(userId, limit) {
    const variant = userId % 2 === 0 ? "dcn" : "wide_and_deep";
    const offset = userId === 42 ? 0 : userId % knownRecommendations.length;
    const ordered = knownRecommendations
      .slice(offset)
      .concat(knownRecommendations.slice(0, offset))
      .slice(0, limit)
      .map((movie, index) => ({
        ...movie,
        reason: variant,
        score: movie.score * (variant === "dcn" ? 1 : 0.982) - index * 0.001,
      }));
    return {
      subject: `user-${userId}`,
      variant,
      strategy: "two_tower_faiss_ranker",
      model_version: MODEL_VERSION,
      cache_hit: false,
      impression_id: createImpression(variant, ordered),
      recommendations: ordered,
    };
  }

  function coldStartResponse(body) {
    const selectedGenres = new Set(body.selected_genres || []);
    const likedMovies = new Set((body.liked_movie_ids || []).map(Number));
    const hasSignals = selectedGenres.size > 0 || likedMovies.size > 0;
    const ranked = catalog
      .filter((movie) => !likedMovies.has(movie.movie_id))
      .map((movie, index) => {
        const overlap = movie.genres.filter((genre) => selectedGenres.has(genre)).length;
        const validatedScore = coldStartScores.get(movie.movie_id) || 0.58 - index * 0.012;
        return { movie, overlap, score: validatedScore + overlap * 0.035 };
      })
      .sort((left, right) => right.overlap - left.overlap || right.score - left.score)
      .slice(0, Math.min(Number(body.limit) || 10, 10))
      .map(({ movie, score }) => ({
        movie_id: movie.movie_id,
        title: movie.title,
        score,
        reason: hasSignals ? "genre_content_popularity" : "diversified_popularity",
        retrieval_score: null,
      }));
    return {
      subject: `session-${body.session_id || "portfolio"}`,
      variant: "cold_start",
      strategy: hasSignals ? "genre_content_popularity" : "diversified_popularity",
      model_version: MODEL_VERSION,
      cache_hit: false,
      impression_id: `portfolio-cold-${crypto.randomUUID()}`,
      recommendations: ranked,
    };
  }

  function clickResponse(body) {
    const impression = impressions.get(body.impression_id);
    const rank = impression?.movieIds.indexOf(Number(body.movie_id)) + 1;
    if (!impression || rank <= 0) throw new Error("Unknown portfolio impression.");
    if (!impression.clicked && metrics[impression.variant]) {
      impression.clicked = true;
      metrics[impression.variant].clicks += 1;
      metrics[impression.variant].discounted_clicks += 1 / Math.log2(rank + 1);
      writeMetrics();
    }
    return { impression_id: body.impression_id, movie_id: Number(body.movie_id), rank };
  }

  async function request(path, options = {}) {
    const url = new URL(path, window.location.origin);
    const method = String(options.method || "GET").toUpperCase();
    const body = options.body ? JSON.parse(options.body) : {};

    if (method === "GET" && url.pathname === "/health") {
      return {
        status: "ok",
        model_version: MODEL_VERSION,
        known_users: 137622,
        indexed_items: 24728,
        cache_healthy: false,
        inference: { backend: "validated_snapshot", ready: true, models: ["dcn", "wide_and_deep"] },
      };
    }

    const knownMatch = url.pathname.match(/^\/recommendations\/(\d+)$/);
    if (method === "GET" && knownMatch) {
      return knownUserResponse(Number(knownMatch[1]), Math.min(Number(url.searchParams.get("limit")) || 10, 10));
    }

    if (method === "POST" && url.pathname === "/recommendations/cold-start") {
      return coldStartResponse(body);
    }

    if (method === "GET" && url.pathname === "/catalog/search") {
      const query = String(url.searchParams.get("q") || "").trim().toLocaleLowerCase();
      const limit = Math.min(Number(url.searchParams.get("limit")) || 6, 10);
      return {
        results: catalog
          .filter((movie) => movie.title.toLocaleLowerCase().includes(query))
          .slice(0, limit),
      };
    }

    if (method === "POST" && url.pathname === "/events/click") {
      return clickResponse(body);
    }

    if (method === "GET" && url.pathname === "/experiments/ranking-model-v1/metrics") {
      return publicMetrics();
    }

    throw new Error(`Portfolio fixture does not implement ${method} ${url.pathname}`);
  }

  window.RecsysPortfolio = {
    get active() {
      return active;
    },
    activate,
    request,
    repositoryUrl: REPOSITORY_URL,
  };

  syncChrome();
})();
