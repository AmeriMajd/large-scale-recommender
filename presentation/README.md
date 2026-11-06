# Final Presentation

This directory contains the final twelve-slide HyperFrames deck, presenter notes,
the real Model Observatory screenshot, and the live demo script.

## Run the deck

The self-contained wrapper only needs a local HTTP server:

```bash
python3 -m http.server 4173 -d presentation
```

Open `http://127.0.0.1:4173`. Use the arrow keys to navigate. Click
**Present**, or press `P`, to open the audience view and presenter notes.

If Node.js 22+ is installed, the HyperFrames presenter server is also available:

```bash
cd presentation
npm run dev
```

## Validate

```bash
cd presentation
npm run check
```

The raw slideshow composition lives in `deck/index.html`; the root
`index.html` is the runnable presenter wrapper.

## Presenting remotely

- Google Meet: share the audience browser tab.
- Zoom: move the audience tab into its own window and share that window.

See [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for the timed narration, live actions,
fallbacks, and likely questions.
