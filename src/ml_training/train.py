# --- IMPORTS ---
import argparse, os, json, time, numpy as np
import torch, torch.nn as nn, torch.optim as optim # Added nn and optim explicitly
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
import mlflow
import mlflow.pytorch # Explicitly import mlflow.pytorch
from model import build_model # Assuming model.py is in the same directory orPYTHONPATH
import matplotlib.pyplot as plt # Needed for plots
import itertools # Needed for CM plot
from utils import plot_confusion_matrix, plot_roc_curve, roc_auc_if_binary

# --- HELPER FUNCTIONS ---

def make_transforms(img_size=224):
    """Creates standard train and validation transforms."""
    # NOTE: Specific transforms for 'condition' vs 'pressure' should ideally be here
    # or passed as args if they differ significantly beyond augmentation.
    # Using the 'condition' transforms from notebook as a base for standard train.
    tfm_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # From 'condition' notebook
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tfm_train, tfm_val

# --- Consolidated Data Loading ---
def setup_dataloaders(data_root, batch_size, img_size, use_kfold=False, num_workers=4):
    """Loads datasets and returns loaders based on K-Fold flag."""
    tfm_train, tfm_val = make_transforms(img_size)
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
         print(f"Warning: Test directory not found: {test_dir}. Final evaluation will be skipped.")
         test_ds = None
    else:
        test_ds = datasets.ImageFolder(test_dir, transform=tfm_val)

    # Determine classes from train_dir reliably
    try:
        classes = datasets.ImageFolder(train_dir).classes
    except Exception as e:
        raise FileNotFoundError(f"Could not determine classes from {train_dir}: {e}")

    if use_kfold:
        print("K-Fold enabled. Combining train and val sets.")
        train_part = datasets.ImageFolder(train_dir, transform=tfm_train)
        if os.path.isdir(val_dir):
            val_part   = datasets.ImageFolder(val_dir,   transform=tfm_train)
            combined_ds = ConcatDataset([train_part, val_part])
            print(f"Combined {len(train_part)} train + {len(val_part)} val = {len(combined_ds)} images for K-Fold.")
        else:
            print(f"Warning: Val dir {val_dir} not found. Using only train set ({len(train_part)} images) for K-Fold.")
            combined_ds = train_part # Fallback to only train if val doesn't exist

        test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None
        return combined_ds, test_loader, classes # Return combined DS for KFold splitting

    else: # Standard train/val split
        print("Using standard Train/Val split.")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Standard split requires validation directory: {val_dir}")

        train_ds = datasets.ImageFolder(train_dir, transform=tfm_train)
        val_ds   = datasets.ImageFolder(val_dir,   transform=tfm_val)
        test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader, classes

def class_weights_from_targets(targets, device):
    """Calculates class weights from a list/array of targets."""
    counts = np.bincount(targets)
    num_classes = len(counts)
    if num_classes < 2: return None # Cannot calculate weights for single class
    # Handle potential zero counts to avoid division by zero
    weights = (1.0 / (counts + 1e-6)) * (counts.sum() / num_classes)
    print(f"Calculated weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32, device=device)

# --- Evaluation Functions (Modified slightly for clarity/robustness) ---
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    y_true, y_pred = [], []
    if loader is None: # Handle case where loader might not exist (e.g., no test set)
        return 0.0, 0.0, np.array([]), np.array([])
        
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # Check if loss_fn is None (might happen if only evaluating)
            if loss_fn:
                 loss = loss_fn(logits, y)
                 loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            
    acc = correct / total if total > 0 else 0
    avg_loss = loss_sum / total if total > 0 else 0
    return acc, avg_loss, np.array(y_true), np.array(y_pred)



def train_standard(args, device):
    """Standard train/val/test loop."""
    print("--- Starting Standard Training ---")
    mlflow.log_params({ # Log relevant args
        "lr_head": args.learning_rate_head,
        "lr_body": args.learning_rate_body,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "dropout": args.dropout # Log dropout
    })

    train_loader, val_loader, test_loader, classes = setup_dataloaders(
        args.data, args.batch_size, args.img_size, use_kfold=False, num_workers=args.num_workers
    )
    num_classes = len(classes)

    # Calculate class weights if needed (condition)
    loss_weights = None
    if args.task == "condition":
        # Get targets from the underlying ImageFolder dataset
        targets = train_loader.dataset.targets
        loss_weights = class_weights_from_targets(targets, device)

    # Build model (dropout enabled, layer4 unfrozen for condition)
    model = build_model(
        num_classes=num_classes, 
        pretrained=True, 
        dropout=args.dropout, # Pass dropout arg
        unfreeze_layer4=True   # Always true for standard training here
    ).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)
    opt = torch.optim.AdamW(
        [{"params": model.layer4.parameters(), "lr": args.learning_rate_body},
         {"params": model.fc.parameters(),     "lr": args.learning_rate_head}],
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=args.patience // 2, verbose=True)

    best_acc = 0.0
    patience_counter = 0
    best_model_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)

        train_loss = run_loss / len(train_loader.dataset)
        val_acc, val_loss, y_true_val, y_pred_val = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {ep:2d}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=ep)
        
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            print(f"  🎉 New best! Saving state... (acc={val_acc:.4f})")
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()} # Save state_dict in CPU memory
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  ⏹️ Early stopping at epoch {ep}")
                break

    # --- After Training Loop ---
    print(f"\n✅ Training complete. Best val_acc={best_acc:.4f}")
    mlflow.log_metric("best_val_acc", best_acc)

    if best_model_state is None:
        raise RuntimeError("Training failed or did not improve: No best model state was saved.")

    # Load best model state for final logging and testing
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Log model to MLflow and register
    print("Logging best model to MLflow and registering...")
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model", 
        registered_model_name=f"tyre-model-{args.task}",
        # Consider adding signature/input_example
    )
    print(f"   Model logged to path: {model_info.artifact_path}")
    print(f"   Model registered as: {model_info.registered_model_name} version {model_info.version if model_info.version else '(new)'}")

    # Final evaluation on test set
    print("\n--- Evaluating best model on Test Set ---")
    test_acc, test_loss, yt_test, yp_test = evaluate(model, test_loader, loss_fn, device)
    if test_loader: # Only report if test set exists
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        mlflow.log_metrics({"final_test_acc": test_acc, "final_test_loss": test_loss})

        cm = confusion_matrix(yt_test, yp_test)
        print("Test Confusion Matrix:\n", cm)
        cm_filename = "confusion_matrix_test.png"
        plot_confusion_matrix(cm, classes, filename=cm_filename)
        mlflow.log_artifact(cm_filename)

        report = classification_report(yt_test, yp_test, target_names=classes)
        print("\nTest Classification Report:\n", report)
        report_filename = "classification_report_test.txt"
        with open(report_filename, "w") as f: f.write(report)
        mlflow.log_artifact(report_filename)

        roc_results = roc_auc_if_binary(model, test_loader, device)
        if roc_results:
            roc_auc, fpr, tpr = roc_results
            print(f"Test AUC: {roc_auc:.4f}")
            mlflow.log_metric("final_test_auc", roc_auc)
            roc_filename = "roc_auc_plot_test.png"
            plot_roc_curve(fpr, tpr, roc_auc, filename=roc_filename)
            mlflow.log_artifact(roc_filename)
    else:
        print("No test set found, skipping final test evaluation.")


def train_pressure_kfold(args, device):
    """K-Fold training loop specifically for pressure dataset."""
    print("--- Starting K-Fold Cross-Validation ---")
    mlflow.log_params({ # Log relevant args
        "k_folds": args.k_folds,
        "learning_rate": args.learning_rate, # Using single LR for pressure head
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "dropout": 0 # Explicitly log no dropout for pressure
    })
    
    # Combine train+val for k-fold, keep test separate
    combined_ds, test_loader, classes = setup_dataloaders(
        args.data, args.batch_size, args.img_size, use_kfold=True, num_workers=args.num_workers
    )
    num_classes = len(classes)

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    cv_scores = []
    best_model_state = None # Overall best state_dict
    best_cv_acc = 0.0

    # K-Fold Loop
    for fold, (tr_idx, va_idx) in enumerate(kf.split(combined_ds), start=1):
        print(f"\n{'='*50}\nFOLD {fold}/{args.k_folds}\n{'='*50}")

        train_subset = Subset(combined_ds, tr_idx)
        val_subset   = Subset(combined_ds, va_idx)

        # Apply correct transforms
        _, tfm_val = make_transforms(args.img_size)
        # We need the underlying ImageFolder paths to create the val_mirror
        # This assumes combined_ds was created from ImageFolders in train/ and val/
        train_dir = os.path.join(args.data, "train")
        val_dir   = os.path.join(args.data, "val")
        
        # Build val_mirror carefully, handle missing val_dir
        val_mirror_datasets = []
        if os.path.isdir(train_dir):
            val_mirror_datasets.append(datasets.ImageFolder(train_dir, transform=tfm_val))
        if os.path.isdir(val_dir):
             val_mirror_datasets.append(datasets.ImageFolder(val_dir, transform=tfm_val))
        
        if not val_mirror_datasets:
             raise FileNotFoundError(f"Cannot create validation mirror: Neither {train_dir} nor {val_dir} found or valid.")
             
        val_mirror = ConcatDataset(val_mirror_datasets)
        val_subset.dataset = val_mirror # Assign mirror with val transforms

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
        val_loader   = DataLoader(val_subset,   batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # Reset model, optimizer, loss
        model = build_model(num_classes=num_classes, pretrained=True, dropout=0, unfreeze_layer4=False).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=args.patience // 2, verbose=True)

        best_fold_acc = 0.0
        patience_ctr = 0
        current_best_fold_state = None

        # --- Epoch Loop for Fold ---
        # Using nested run for fold to keep metrics separate
        with mlflow.start_run(run_name=f"Fold_{fold}", nested=True):
             mlflow.log_param("fold", fold)
             for ep in range(1, args.epochs+1):
                # train
                model.train(); run_loss = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
                    run_loss += loss.item() * xb.size(0)
                tr_loss = run_loss / len(train_subset)

                # val
                acc, vloss, _, _ = evaluate(model, val_loader, loss_fn, device)
                mlflow.log_metrics({"train_loss": tr_loss, "val_loss": vloss, "val_acc": acc}, step=ep)
                sched.step(acc)

                if acc > best_fold_acc:
                    best_fold_acc = acc; patience_ctr = 0
                    print(f"  🎉 New fold best! (acc={acc:.4f})")
                    current_best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= args.patience:
                        print(f"  ⏹️ Early stopping triggered at epoch {ep}")
                        break # Stop training this fold
             
             # Log best fold accuracy at the end of the fold run
             mlflow.log_metric("best_fold_val_acc", best_fold_acc)

        # --- End of Fold ---
        cv_scores.append(best_fold_acc)
        # Log best fold accuracy also to the PARENT run for easier comparison
        mlflow.log_metric(f"fold_{fold}_best_val_acc", best_fold_acc, run_id=mlflow.active_run().info.run_id) 

        print(f"Fold {fold} completed. Best accuracy: {best_fold_acc:.4f}")

        # Track OVERALL best model across folds
        if current_best_fold_state and best_fold_acc > best_cv_acc: # Check state exists
            best_cv_acc = best_fold_acc
            best_model_state = current_best_fold_state
            print(f"   🏆 New OVERALL best model found in Fold {fold}!")
            # Log overall best score to PARENT run
            mlflow.log_metric("best_overall_cv_acc", best_cv_acc, run_id=mlflow.active_run().info.run_id)

    # --- AFTER ALL K-FOLDS COMPLETE (in parent run) ---
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"\n{'='*50}\nCROSS-VALIDATION RESULTS\n{'='*50}")
    print(f"Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"Mean CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    mlflow.log_metric("mean_cv_accuracy", mean_cv_score)
    mlflow.log_metric("std_cv_accuracy", std_cv_score)

    if best_model_state is None:
        raise RuntimeError("Training failed: No best model state was saved during cross-validation.")

    # --- Load, Log, and Test the Overall Best Model ---
    print(f"\nLoading overall best model state (CV Acc: {best_cv_acc:.4f}) for final logging and evaluation...")
    final_best_model = build_model(num_classes=len(classes), pretrained=True, dropout=0, unfreeze_layer4=False).to(device)
    final_best_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    print("Logging best model from CV to MLflow and registering...")
    artifact_path = "best_cv_model"
    registered_model_name = f"tyre-model-{args.task}-cv"
    model_info = mlflow.pytorch.log_model(
        pytorch_model=final_best_model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
    print(f"   Model logged to path: {model_info.artifact_path}")
    print(f"   Model registered as: {model_info.registered_model_name} version {model_info.version if model_info.version else '(new)'}")

    # Evaluate on test set
    if test_loader:
        print("\n--- Evaluating best model on Test Set ---")
        loss_fn_test = torch.nn.CrossEntropyLoss()
        test_acc, test_loss, yt_test, yp_test = evaluate(final_best_model, test_loader, loss_fn_test, device)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        mlflow.log_metrics({"final_test_acc": test_acc, "final_test_loss": test_loss})

        cm = confusion_matrix(yt_test, yp_test)
        print("Test Confusion Matrix:\n", cm)
        cm_filename = "confusion_matrix_test.png"
        plot_confusion_matrix(cm, classes, filename=cm_filename)
        mlflow.log_artifact(cm_filename)

        report = classification_report(yt_test, yp_test, target_names=classes)
        print("\nTest Classification Report:\n", report)
        report_filename = "classification_report_test.txt"
        with open(report_filename, "w") as f: f.write(report)
        mlflow.log_artifact(report_filename)

        roc_results = roc_auc_if_binary(final_best_model, test_loader, device)
        if roc_results:
            roc_auc, fpr, tpr = roc_results
            print(f"Test AUC: {roc_auc:.4f}")
            mlflow.log_metric("final_test_auc", roc_auc)
            roc_filename = "roc_auc_plot_test.png"
            plot_roc_curve(fpr, tpr, roc_auc, filename=roc_filename)
            mlflow.log_artifact(roc_filename)
    else:
        print("\nNo test set found, skipping final test evaluation.")

    print(f"\n✅ K-Fold Training and Logging complete!")


def main():
    ap = argparse.ArgumentParser()
    # --- Arguments ---
    # Task Selection
    ap.add_argument("--task", choices=["condition","pressure"], required=True, help="Which tyre task to train.")
    # Data Input (Expects path mounted by AML)
    ap.add_argument("--data", required=True, help="Root folder containing train/val/test subdirectories.")
    # Model Output (Used locally)
    # ap.add_argument("--out", default="outputs", help="Local output directory for model.") 
    
    # Training Hyperparameters
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--learning_rate", type=float, default=1e-3, help="LR for pressure head")
    ap.add_argument("--learning_rate_head", type=float, default=5e-4, help="LR for condition head")
    ap.add_argument("--learning_rate_body", type=float, default=1e-6, help="LR for condition body (layer4)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4, help="Patience for early stopping")
    ap.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for condition model head (0 = disabled)")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # K-Fold Control
    ap.add_argument("--k_folds", type=int, default=0, help="Number of K-Folds. 0 disables K-Fold, >1 enables (for pressure).")
    
    args = ap.parse_args()

    # --- Argument Validation ---
    if args.task == "pressure" and args.k_folds <= 1:
        print("Info: Running 'pressure' task without K-Fold (k_folds=0 or 1). Using standard train/val split.")
        args.use_kfold = False
    elif args.task == "pressure" and args.k_folds > 1:
         print(f"Info: Running 'pressure' task with {args.k_folds}-Fold CV.")
         args.use_kfold = True
    elif args.task == "condition" and args.k_folds > 1:
         print("Warning: K-Fold specified for 'condition' task, but standard train/val split will be used.")
         args.use_kfold = False
    else: # condition and k_folds=0 or 1
        args.use_kfold = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- MLflow Setup ---
    # AML automatically sets MLflow tracking URI, but good practice locally too
    mlflow.set_experiment(f"Tyre_Model_{args.task}") 

    # --- Main Training Run ---
    # Using a single parent run, K-Fold folds will be nested runs
    with mlflow.start_run() as parent_run: 
        print(f"Starting MLflow Run ID: {parent_run.info.run_id}")
        mlflow.log_params(vars(args)) # Log all args

        if args.task == "pressure" and args.use_kfold:
            train_pressure_kfold(args, device)
        else: # Condition or Pressure without K-Fold
             # Determine specific LR/WD for standard training based on task
             lr_h = args.learning_rate_head if args.task == "condition" else args.learning_rate
             lr_b = args.learning_rate_body if args.task == "condition" else 0 # No body LR needed if layer4 frozen
             wd = args.weight_decay
             
             train_standard(args, device)
        
        print(f"\n✅ Training run {parent_run.info.run_id} completed.")
        print(f"   Check MLflow UI for detailed results and artifacts.")

if __name__ == "__main__":
    main()