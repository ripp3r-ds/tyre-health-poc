# Tyre Health Detection (Proof of Concept)

## Objective
This project is a proof-of-concept for a lightweight machine learning system. It aims to classify both the condition (e.g., worn) and pressure status (e.g., flat) of a tyre from a single image. The intended application is as a prototype for automated safety checks at locations like toll gates or parking lots.

---

## Technology Stack
* **Model Training:** PyTorch, initially on a local GPU, with plans to use Azure ML for managed training jobs.
* **Model Deployment:** Azure ML Managed Online Endpoint.
* **API Layer:** FastAPI application hosted on Azure Container Apps.
* **Frontend:** A minimal HTML/JS interface for testing image uploads and viewing predictions.
* **Storage:** Azure Blob Storage for datasets and logs.

---

## Project Structure
* `src/`: Core Python source code for model training, inference logic, and utilities.
* `score/`: Deployment-related scripts and artifacts required by Azure ML.
* `api/`: The FastAPI microservice that serves as a wrapper for the ML endpoint.
* `ui/`: Optional static web interface for demo purposes.
* `azure/`: Infrastructure-as-Code (IaC) configurations for deploying Azure ML and Container App resources.
* `data/`: Local directory for datasets (not checked into git). Contains raw, processed, and split data.
* `models/`: Saved model weights (e.g., `.pth` files) after training.

---

## Datasets
This project uses a combination of publicly available datasets to train the classification models.

**Sources:**
1.  [Tire Tread Dataset (Roboflow)](https://universe.roboflow.com/mark-aft7n/tire-tread)
2.  [Tyre Quality Classification (GTS.ai)](https://gts.ai/dataset-download/tyre-quality-classification-dataset-for-ai-analysis/)
3.  [Full vs Flat Tire Images (Kaggle)](https://www.kaggle.com/datasets/rhammell/full-vs-flat-tire-images)

**Target Classification Tasks:**
* **Condition:** Classifying the tyre tread as 'Normal' or 'Worn'.
* **Pressure:** Classifying the tyre pressure as 'Full' or 'Flat/Underinflated'.

---

## Local Setup
To run this project locally, follow these steps:

```bash
# Clone the repository
git clone <repo_url>
cd tyre-health-poc

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the required dependencies
pip install -r api/requirements.txt
