# Tire Image Dataset Preprocessing Scripts

---

## Overview

This project contains two Python scripts to process and organize tire image datasets for model training.

* **Script 1 (Pressure):** Splits images from `full.class` and `flat.class` folders into `train`, `val`, and `test` sets.
* **Script 2 (Condition):** Reads a Roboflow dataset export, re-labels images based on a CSV file into `good` and `worn` categories, and organizes them.

---

## Setup

* **Dependencies:** The scripts require `python-dotenv`.
    ```bash
    pip install python-dotenv
    ```
* **Environment File:** Create a `.env` file in the root directory to define the dataset source paths.

    **`.env.example`**
    ```ini
    # Path to the "tire-pressure-dataset"
    src_kaggle="/path/to/your/tire-pressure-dataset"

    # Path to the Roboflow "Tire Tread" dataset
    src_roboflow="/path/to/your/roboflow-tire-tread-dataset"
    ```

---

## 1. Pressure Dataset Script (`process_pressure_dataset.py`)

This script splits the "pressure" dataset according to a defined ratio.

* **Input Structure:**
    ```
    <src_kaggle>/
    ├── full.class/
    │   ├── img1.jpg
    │   └── ...
    └── flat.class/
        ├── img2.jpg
        └── ...
    ```
* **How to Run:**
    ```bash
    python process_pressure_dataset.py
    ```
* **Output Structure:**
    ```
    data/raw/pressure/
    ├── train/
    │   ├── full/
    │   └── flat/
    ├── val/
    │   ├── full/
    │   └── flat/
    └── test/
        ├── full/
        └── flat/
    ```

---

## 2. Condition Dataset Script (`process_condition_dataset.py`)

This script reads the Roboflow export, parses the `_classes.csv` file, and copies images into new `good` and `worn` categories.

* **Input Structure:**
    ```
    <src_roboflow>/
    ├── train/
    │   ├── _classes.csv
    │   ├── img1.jpg
    │   └── ...
    ├── valid/
    │   ├── _classes.csv
    │   └── ...
    └── test/
        ├── _classes.csv
        └── ...
    ```
* **How to Run:**
    ```bash
    python process_condition_dataset.py
    ```
* **Output Structure:**
    ```
    data/raw/condition/
    ├── train/
    │   ├── good/
    │   └── worn/
    ├── val/
    │   ├── good/
    │   └── worn/
    └── test/
        ├── good/
        └── worn/
    ```