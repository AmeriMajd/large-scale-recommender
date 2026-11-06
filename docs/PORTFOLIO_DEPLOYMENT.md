# Portfolio Deployment

The Model Observatory supports two runtime modes from the same `app/` directory:

- FastAPI mode uses the real local recommendation endpoints.
- Portfolio mode uses validated frozen responses and browser-local interactions.

Portfolio mode activates automatically on `*.pages.dev`, when the backend is
unavailable, or when the page is opened with `?portfolio=1`. The interface
displays a persistent disclosure banner and never describes fixture events as
production traffic.

## Public deployment

The production portfolio is available at:

<https://large-scale-recommender.pages.dev>

The current Cloudflare project uses Direct Upload. Redeploy it with:

```bash
npx wrangler pages deploy app --project-name large-scale-recommender --branch main
```

Pushing to GitHub does not automatically redeploy this Direct Upload project.

## Git-integrated alternative

Cloudflare does not convert an existing Direct Upload project to Git
integration. To use automatic deployment on every push, create a separate Pages
project and configure it as follows:

1. In Cloudflare, open **Workers & Pages** and create a Pages application.
2. Connect `AmeriMajd/large-scale-recommender`.
3. Select the `main` production branch.
4. Use no framework preset.
5. Set the build command to `exit 0`.
6. Set the build output directory to `app`.
7. Deploy.

The resulting address will be similar to
`https://large-scale-recommender.pages.dev`.

## Direct upload

After authenticating Wrangler, deploy the static directory directly:

```bash
npx wrangler pages deploy app --project-name large-scale-recommender
```

Cloudflare Pages reads `app/_headers` during deployment. No dataset, model
binary, secret, Redis instance, or inference server is uploaded.

## Local static verification

```bash
python3 -m http.server 4173 -d app
```

Open `http://127.0.0.1:4173/?portfolio=1`.

For the real backend, continue to use `make serve` and open
`http://127.0.0.1:8000` without the portfolio query parameter.
