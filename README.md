# TireCheck AI

Quick, AI-assisted checks to spot common tire issues before they get expensive. Built for demos — not a replacement for professional inspection.

## Tech stack
- Frontend: React + TypeScript + Vite + Tailwind
- API: FastAPI (Python) calling Azure ML Managed Online Endpoint
- Training: PyTorch + MLflow
- Infra: Azure ML, Azure Container Apps, Azure Static Web Apps

## Project structure (final)
```
tyre-health-poc/
├── frontend/                      # React app (Vite)
├── src/
│   ├── api/                      # FastAPI app (src/api/main.py)
│   └── ml_training/              # Training code and scripts
├── aml/                           # Azure ML job/env specs
│   ├── environment/
│   └── endpoints/
├── score/                         # Azure ML scoring scripts/env
├── data-preprocessing/            # Dataset preparation helpers
├── notebooks/                     # Experiment notebooks + artifacts
└── README.md
```

## Run locally

### 1) Backend API
Create `src/api/.env` with:

```
AML_ENDPOINT=https://<your-aml-endpoint>.azurewebsites.net/score
AML_KEY=<your-aml-key>
```

Install and run:
```bash
cd src/api
python -m venv .venv && . .venv/Scripts/activate  # on Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Frontend
Set the API base URL in `frontend/.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```
Run:
```bash
cd frontend
npm install
npm run dev
```

## Training
See `src/ml_training/README.md` for environment setup, scripts, and MLflow usage.

## What should not be committed
- `venv/`, `.venv/`, `node_modules/`
- `mlruns/`, `outputs/`, `notebooks/models/`, `notebooks/artifacts/`
- raw datasets under `data/`

Consider adding these to `.gitignore` if not already ignored.

## Deployment overview
- Azure ML hosts the model endpoint; scoring scripts live in `score/`
- FastAPI runs in Azure Container Apps; set `AML_ENDPOINT` and `AML_KEY` as secrets
- Frontend deploys to Azure Static Web Apps; set `VITE_API_BASE_URL` to the container app URL

See the “Deployment notes” at the end of this document for common gotchas.

## License
MIT — see `LICENSE`.

## Deployment notes (Azure SWA + ACA)
1) Configure CORS on the FastAPI app (already enabled for localhost). For cloud, allow the SWA domain.
2) Expose container port 8000 and ensure the ACA ingress is set to external.
3) Set secrets in ACA: `AML_ENDPOINT`, `AML_KEY`.
4) In SWA, add `VITE_API_BASE_URL` as an environment variable (or build-time secret) and rebuild.
5) If using custom domains, update allowed origins and HTTPS-only settings.

Troubleshooting:
- 403/401 from AML: verify `AML_KEY` and correct endpoint URL path (`/score`).
- CORS preflight failures: confirm ACA returns 200 to `OPTIONS` and `Access-Control-Allow-Origin` matches SWA.
- Mixed content errors: ensure both SWA and ACA are HTTPS.
