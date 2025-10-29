def roc_auc_if_binary(model, loader, device):
    model.eval()
    y_true, y_proba1 = [], []
    num_classes = -1
    
    if loader is None: return None
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prob = torch.softmax(model(x), dim=1)
            if num_classes == -1: num_classes = prob.shape[1] # Get num_classes from first batch
            
            # Only proceed if binary
            if num_classes != 2: return None

            y_true.extend(y.cpu().numpy().tolist())
            y_proba1.extend(prob[:, 1].cpu().numpy().tolist()) # Probability of class 1

    if not y_true or len(np.unique(y_true)) < 2: # Check if only one class present in labels
        print("Warning: ROC AUC calculation requires labels for at least 2 classes.")
        return None 
        
    fpr, tpr, _ = roc_curve(y_true, y_proba1)
    roc_auc = auc(fpr, tpr)
    return float(roc_auc), fpr.tolist(), tpr.tolist()

# --- Plotting Functions ---
def plot_confusion_matrix(cm, classes, filename="confusion_matrix.png"):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right") # Improve label alignment
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # Important for non-interactive jobs
    print(f"Saved confusion matrix to {filename}")

def plot_roc_curve(fpr, tpr, roc_auc, filename="roc_auc_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved ROC plot to {filename}. AUC: {roc_auc:.4f}")

