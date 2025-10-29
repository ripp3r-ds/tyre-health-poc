import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, classification_report

from model import build_model
from dataloaders import load_standard_split
from utils import evaluate, roc_auc_if_binary, plot_confusion_matrix, plot_roc_curve


def class_weights_from_loader(train_loader, device, classes):
    # Collect targets from underlying dataset
    targets = []
    try:
        targets = train_loader.dataset.targets  # ImageFolder exposes .targets
    except AttributeError:
        # Fall back to iterating once (slower but safe)
        for _, y in train_loader:
            targets.extend(y.numpy().tolist())
    counts = np.bincount(targets, minlength=len(classes))
    total = counts.sum()
    weights = np.zeros(len(classes), dtype=np.float32)
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = (1.0 / c) * (total / len(classes))
        else:
            weights[i] = 0.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser(description="Train CONDITION model (standard split, early-stop on val loss)")
    ap.add_argument("--data", required=True, help="Root folder containing train/val/test")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--lr_body", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlflow.set_experiment("Tyre_Model_condition")

    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))
        train_loader, val_loader, test_loader, classes = load_standard_split(
            args.data, args.batch_size, args.img_size, args.num_workers, task="condition"
        )

        model = build_model(
            num_classes=len(classes), pretrained=True,
            dropout=args.dropout, unfreeze_layer4=True
        ).to(device)

        weights = class_weights_from_loader(train_loader, device, classes)
        loss_fn = nn.CrossEntropyLoss(weight=weights if weights is not None else None)

        optimizer = optim.AdamW([
            {"params": model.layer4.parameters(), "lr": args.lr_body},
            {"params": model.fc.parameters(),     "lr": args.lr_head},
        ], weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 2))

        best_state, best_val_loss, patience_ctr = None, float("inf"), 0

        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            # --- Train ---
            model.train(); run_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                run_loss += float(loss.item()) * xb.size(0)
            train_loss = run_loss / len(train_loader.dataset)

            # --- Validate ---
            val_acc, val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
            scheduler.step(val_loss)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=ep)
            print(f"Epoch {ep}/{args.epochs} | tr_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | {time.time()-t0:.1f}s")

            # Early stopping on val_loss
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"Early stopping at epoch {ep}")
                    break

        # Load best
        if best_state is None:
            raise RuntimeError("No best state saved — training did not run?")
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Log/register
        try:
            info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                registered_model_name="tyre-model-condition",
            )
            mlflow.log_param("registered_model_name", getattr(getattr(info, "registered_model_version", None), "name", "None"))
            mlflow.log_param("registered_model_version", getattr(getattr(info, "registered_model_version", None), "version", "None"))
        except Exception as e:
            # Local MLflow (file store) may not have Registry — allow run to continue
            mlflow.log_param("registration_warning", str(e))
            print(f"Model registration skipped/failed: {e}")

        # Final test
        if test_loader is not None:
            test_acc, test_loss, y_true, y_pred = evaluate(model, test_loader, loss_fn, device)
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
            roc = roc_auc_if_binary(model, test_loader, device)
            if roc is not None:
                auc_val, fpr, tpr = roc
                mlflow.log_metric("test_auc", auc_val)
                plot_roc_curve(fpr, tpr, auc_val, filename="roc_test.png")
                mlflow.log_artifact("roc_test.png")


if __name__ == "__main__":
    main()