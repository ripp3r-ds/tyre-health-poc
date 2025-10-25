
# 🛞 Tyre Health Detection POC

**Goal:**  
A lightweight ML-based system that classifies tyre condition and pressure status from a single image, designed as a prototype for toll-gate or parking-lot safety checks.

---

## 🚀 Tech Stack
- **Training:** PyTorch (local GPU) → Azure ML (managed jobs later)
- **Deployment:** Azure ML Managed Online Endpoint
- **API:** FastAPI hosted on Azure Container Apps
- **UI:** Minimal HTML/JS for image upload & prediction
- **Storage:** Azure Blob Storage (dataset & logs)

---

## 📁 Project Structure
- `src/` → training and inference code  
- `score/` → deployment artifacts for Azure ML  
- `api/` → FastAPI microservice to call the ML endpoint  
- `ui/` → static web interface (optional)  
- `azure/` → IaC configs for AML and ACA  
- `data/` → local datasets (raw, processed, splits)  
- `models/` → trained weights  

---

## 🧠 Datasets
1. [Tire Tread Dataset (Roboflow)](https://universe.roboflow.com/mark-aft7n/tire-tread)
2. [Tyre Quality Classification (GTS.ai)](https://gts.ai/dataset-download/tyre-quality-classification-dataset-for-ai-analysis/)
3. [Full vs Flat Tire Images (Kaggle)](https://www.kaggle.com/datasets/rhammell/full-vs-flat-tire-images)

Tasks:
- `condition`: Normal vs Worn
- `pressure`: Full vs Flat/Underinflated

---

## 🧩 Local Setup
```bash
# clone and setup
git clone <repo_url>
cd tyre-health-poc
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate (Windows)
pip install -r api/requirements.txt

