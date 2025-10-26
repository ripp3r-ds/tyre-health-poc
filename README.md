# Tyre Health Detection (Proof of Concept)

## 🎯 Objective
This project is a proof-of-concept for a lightweight machine learning system that classifies both the **condition** (good vs worn) and **pressure status** (full vs flat) of a tyre from a single image. The intended application is automated safety checks at locations like toll gates or parking lots.

## 🏗️ Technology Stack
* **Model Training:** PyTorch with transfer learning and custom CNNs
* **Experiment Tracking:** MLflow for model versioning and metrics
* **Model Deployment:** Azure ML Managed Online Endpoint
* **API Layer:** FastAPI application hosted on Azure Container Apps
* **Frontend:** HTML/JS interface for testing image uploads
* **Storage:** Azure Blob Storage for datasets and model artifacts

## 📁 Project Structure
```
tyre-health-poc/
├── api/                    # FastAPI microservice
│   ├── app.py             # Main API application
│   ├── requirements.txt   # API dependencies
│   └── Dockerfile         # Container configuration
├── azure/                 # Infrastructure as Code
│   ├── containerapp.yml   # Container App configuration
│   ├── ml_deploy.yml      # ML deployment configuration
│   └── infra.md           # Infrastructure documentation
├── data/                  # Dataset storage (not in git)
│   └── raw/
│       ├── condition/     # Tyre condition dataset
│       └── pressure/      # Tyre pressure dataset
├── data-preprocessing/    # Data preparation scripts
│   ├── kaggle_data_process.py
│   ├── roboflow_data_process.py
│   └── README.md
├── notebooks/             # Training notebooks
│   ├── training_condition.ipynb
│   ├── training_pressure.ipynb
│   ├── models/            # Trained model weights
│   └── artifacts/         # Training artifacts
├── score/                 # Azure ML deployment artifacts
│   ├── score.py           # Inference script
│   ├── conda.yml          # Environment specification
│   └── inference_config.json
├── src/                   # Core source code
│   ├── model.py           # Model definitions
│   ├── train.py           # Training utilities
│   ├── inference.py       # Inference utilities
│   └── utils.py           # Helper functions
└── ui/                    # Web interface
    └── static_webapp/
        ├── index.html
        └── script.js
```

## 📊 Datasets

### **Condition Dataset (Tyre Tread)**
- **Source:** [Roboflow Tire Tread Dataset](https://universe.roboflow.com/mark-aft7n/tire-tread)
- **Classes:** `good`, `worn`
- **Size:** ~8,000 images
- **Split:** train/val/test (80/10/10)
- **Model:** ResNet18 with transfer learning

### **Pressure Dataset (Tyre Pressure)**
- **Source:** [Kaggle Full vs Flat Tire Images](https://www.kaggle.com/datasets/rhammell/full-vs-flat-tire-images)
- **Classes:** `full`, `flat`
- **Size:** ~500 images (small dataset)
- **Split:** train/val/test (78/12/10)
- **Model:** Custom CNN optimized for small datasets

## 🚀 Local Setup

### **Prerequisites**
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### **Installation**
```bash
# Clone the repository
git clone <repo_url>
cd tyre-health-poc

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r api/requirements.txt
```



## 🏃‍♂️ Quick Start

### **1. Data Preparation**
```bash
# Process datasets
cd data-preprocessing
python kaggle_data_process.py    # Process pressure dataset
python roboflow_data_process.py  # Process condition dataset
```

### **2. Model Training**
```bash
# Train condition model
jupyter notebook notebooks/training_condition.ipynb

# Train pressure model  
jupyter notebook notebooks/training_pressure.ipynb
```

### **3. Model Evaluation**
- **Condition Model**: ~90% accuracy on validation set
- **Pressure Model**: ~75-85% accuracy (realistic for small dataset)
- **Models saved**: `notebooks/models/`

### **4. API Testing**
```bash
# Start API server
cd api
python app.py

# Test with sample image
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_tyre.jpg"
```

## 📈 Model Performance

### **Condition Classification (ResNet18)**
- **Validation Accuracy**: ~90%
- **Classes**: good, worn
- **Training Time**: ~30 minutes (GPU)
- **Model Size**: ~45MB

### **Pressure Classification (Custom CNN)**
- **Cross-Validation**: ~75-85%
- **Classes**: full, flat
- **Training Time**: ~15 minutes (GPU)
- **Model Size**: ~10MB
- **Optimized for**: Small datasets, safety-critical applications

## Azure Deployment

### **Prerequisites**
- Azure subscription
- Azure ML workspace
- Azure Container Registry

### **Deploy to Azure**
```bash
# Deploy ML models
az ml online-endpoint create -f azure/ml_deploy.yml

# Deploy API service
az containerapp create -f azure/containerapp.yml
```

## Key Features

### **Condition Detection**
- **Transfer Learning**: ResNet18 pre-trained on ImageNet
- **Class Imbalance**: Handled with weighted loss
- **Augmentation**: Horizontal flip, color jitter
- **Architecture**: Frozen backbone + trainable head

### **Pressure Detection**
- **Transfer Learning**: ResNet18 pre-trained on ImageNet
- **Class Imbalance**: Handled with weighted loss
- **Augmentation**: Horizontal flip, color jitter
- **Architecture**: Frozen backbone + trainable head
## 📊 Experiment Tracking

All training runs are tracked with MLflow:
- **Metrics**: Loss, accuracy, precision, recall
- **Parameters**: Learning rates, batch sizes, model configs
- **Artifacts**: Models, confusion matrices, ROC curves
- **UI**: Access at `http://localhost:5000`

## 🛠️ Development

### **Adding New Models**
1. Create model class in `src/model.py`
2. Add training script in `notebooks/`
3. Update inference in `score/score.py`
4. Test with API

### **Data Pipeline**
1. Raw data → `data-preprocessing/` scripts
2. Processed data → `data/raw/`
3. Training → `notebooks/`
4. Models → `notebooks/models/`

## 📝 Notes


- **Overfitting**: Controlled with early stopping and regularization
- **Safety Priority**: Pressure model optimized for recall (flat tyre detection)
- **Scalability**: Designed for Azure ML deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
