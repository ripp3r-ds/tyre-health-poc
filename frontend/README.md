# Frontend (React + TypeScript + Vite)

This is the UI for TireCheck AI. It talks to the FastAPI backend and provides animated flows for model selection, guidance, upload, and results.

## Env vars

Create a `.env` in this folder:

```
VITE_API_BASE_URL=http://localhost:8000
```

## Run locally

```bash
npm install
npm run dev
# open the printed localhost:5173 URL
```

## Build

```bash
npm run build
npm run preview
```

## Design language

- TailwindCSS, dark gradient background
- Subtle entrance animations (animate-in, slide/zoom)
- Responsive grid and tightened vertical rhythm

Pages live in `src/pages/` and share UI primitives from `src/components/ui/`.
