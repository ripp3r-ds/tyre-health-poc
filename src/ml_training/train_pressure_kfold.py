import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, ConcatDataset
from torchvision import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
import mlflow.pytorch

from model import build_model
from dataloaders import make_transforms_pressure
from utils import evaluate, roc_auc_if_binary, plot_confusion_matrix, plot_roc_curve


def build_trainval_datasets(root: Path, img_size: int):
    """Return (trainval_trainTfm, trainval_evalTfm, classes)."""
    train_dir = root / "train"
    val_dir   = root / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing val dir: {val_dir}")

    tfm_train, tfm_eval = make_transforms_pressure(img_size)

    # For class names
    tmp = datasets.ImageFolder(str(train_dir))
    classes = tmp.classes
    if len(classes) < 2:
        raise ValueError(f"Expected at least 2 classes; got {len(classes)} in {train_dir}")

    ds_train_trainTfm = datasets.ImageFolder(str(train_dir), transform=tfm_train)
    ds_val_trainTfm   = datasets.ImageFolder(str(val_dir),   transform=tfm_train)
    ds_train_evalTfm  = datasets.ImageFolder(str(train_dir), transform=tfm_eval)
    ds_val_evalTfm    = datasets.ImageFolder(str(val_dir),   transform=tfm_eval)

    base_trainTfm = ConcatDataset([ds_train_trainTfm, ds_val_trainTfm])
    base_evalTfm  = ConcatDataset([ds_train_evalTfm,  ds_val_evalTfm])
    return base_trainTfm, base_evalTfm, classes


def build_test_loader(root: Path, img_size: int, batch_size: int, num_workers: int, pin_memory: bool):
    test_dir = root / "test"
    if not test_dir.is_dir():
        return None
    _, tfm_eval = make_transforms_pressure(img_size)
    test_ds = datasets.ImageFolder(str(test_dir), transform=tfm_eval)
    return torch.utils.data.DataLoader(test_ds, batch_size=batch_size*2, shuffle=False,
                                       num_workers=num_workers, pin_memory=pin_memory)


def main():
    ap = argparse.ArgumentParser(description="Pressure K-Fold CV (early-stop on val loss, CPU/GPU aware)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = torch.cuda.is_available()

    mlflow.set_experiment("Tyre_Model_pressure_cv")

    root = Path(args.data)
    base_trainTfm, base_evalTfm, classes = build_trainval_datasets(root, args.img_size)
    test_loader = build_test_loader(root, args.img_size, args.batch_size, args.num_workers, pin_memory)

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    best_overall_state = None
    best_overall_val_loss = float("inf")
    fold_losses = []

    with mlflow.start_run() as parent_run:
        mlflow.log_params(vars(args))

        for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(base_trainTfm))), start=1):
            print(f"\n{'='*40}\nFOLD {fold}/{args.k_folds}\n{'='*40}")
            train_subset = Subset(base_trainTfm, tr_idx)
            val_subset   = Subset(base_evalTfm,  va_idx)  # eval transforms for validation

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.num_workers, pin_memory=pin_memory)
            val_loader   = torch.utils.data.DataLoader(val_subset,   batch_size=args.batch_size*2, shuffle=False,
                                                       num_workers=args.num_workers, pin_memory=pin_memory)

            model = build_model(num_classes=len(classes), pretrained=True, dropout=0.0, unfreeze_layer4=False).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                              patience=max(1, args.patience // 2))

            best_state, best_val_loss, patience_ctr = None, float("inf"), 0

            with mlflow.start_run(run_name=f"Fold_{fold}", nested=True) as fold_run:
                for ep in range(1, args.epochs + 1):
                    t0 = time.time()
                    # Train
                    model.train(); run_loss = 0.0
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                        loss.backward()
                        optimizer.step()
                        run_loss += float(loss.item()) * xb.size(0)
                    train_loss = run_loss / len(train_subset)

                    # Validate
                    val_acc, val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
                    scheduler.step(val_loss)

                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    }, step=ep)
                    print(f"Fold {fold} | Epoch {ep}/{args.epochs} | tr_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | {time.time()-t0:.1f}s")

                    # Early stop on val_loss
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        patience_ctr = 0
                    else:
                        patience_ctr += 1
                        if patience_ctr >= args.patience:
                            print(f"Early stopping at epoch {ep}")
                            break

                mlflow.log_metric("fold_best_val_loss", best_val_loss)

            fold_losses.append(best_val_loss)
            if best_state is not None and best_val_loss < best_overall_val_loss:
                best_overall_val_loss = best_val_loss
                best_overall_state = best_state

        # After all folds
        mlflow.log_metric("cv_mean_val_loss", float(np.mean(fold_losses)))
        mlflow.log_metric("cv_std_val_loss", float(np.std(fold_losses)))

        if best_overall_state is None:
            raise RuntimeError("K-Fold finished but no best model state was captured.")

        # Rebuild model and load best
        final_model = build_model(num_classes=len(classes), pretrained=True, dropout=0.0, unfreeze_layer4=False).to(device)
        final_model.load_state_dict({k: v.to(device) for k, v in best_overall_state.items()})

        # Log/register best CV model
        try:
            info = mlflow.pytorch.log_model(
                pytorch_model=final_model,
                artifact_path="best_cv_model",
                registered_model_name="tyre-model-pressure-cv",
            )
            # Save registered model info if available
            name = getattr(getattr(info, "registered_model_version", None), "name", "None")
            ver  = getattr(getattr(info, "registered_model_version", None), "version", "None")
            mlflow.log_param("registered_model_name", name)
            mlflow.log_param("registered_model_version", ver)
        except Exception as e:
            mlflow.log_param("registration_warning", str(e))
            print(f"Model registration skipped/failed: {e}")

        # Evaluate on test set (if present)
        if test_loader is not None:
            loss_fn = nn.CrossEntropyLoss()
            test_acc, test_loss, y_true, y_pred = evaluate(final_model, test_loader, loss_fn, device)
            mlflow.log_metrics({"test_acc": test_acc, "test_loss": test_loss})
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, classes, filename="cm_test.png")
            mlflow.log_artifact("cm_test.png")
            try:
                report = classification_report(y_true, y_pred, target_names=classes)
                with open("classification_report_test.txt", "w") as f:
                    f.write(report)
                mlflow.log_artifact("classification_report_test.txt")
            except Exception:
                pass
            roc = roc_auc_if_binary(final_model, test_loader, device)
            if roc is not None:
                auc_val, fpr, tpr = roc
                mlflow.log_metric("test_auc", auc_val)
                plot_roc_curve(fpr, tpr, auc_val, filename="roc_test.png")
                mlflow.log_artifact("roc_test.png")


if __name__ == "__main__":
    main()