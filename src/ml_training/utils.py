from typing import Tuple, List, Optional
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import itertools

# --- Evaluation helpers ---

def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: str) -> Tuple[float, float, List[int], List[int]]:
    """Return (acc, loss, y_true, y_pred)."""
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.size(0)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    mean_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return acc, mean_loss, y_true, y_pred


def roc_auc_if_binary(model: torch.nn.Module,
                      loader: Optional[torch.utils.data.DataLoader],
                      device: str):
    """Return (roc_auc, fpr, tpr) if binary; otherwise None."""
    if loader is None:
        return None
    model.eval()
    y_true, y_proba1 = [], []
    num_classes = -1

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            prob = torch.softmax(model(xb), dim=1)
            if num_classes == -1:
                num_classes = prob.shape[1]
            if num_classes != 2:
                return None
            y_true.extend(yb.cpu().numpy().tolist())
            y_proba1.extend(prob[:, 1].cpu().numpy().tolist())

    if not y_true or len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_proba1)
    roc_auc = auc(fpr, tpr)
    return float(roc_auc), fpr.tolist(), tpr.tolist()


# --- Plotting ---

def plot_confusion_matrix(cm, classes, filename="confusion_matrix.png"):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, filename="roc_auc_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()