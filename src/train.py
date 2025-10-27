import argparse, os, json, time, numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
import mlflow
from model import build_model

def make_transforms(img_size=224):
    tfm_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

def make_loaders(root, batch_size, img_size, use_combined=False):
    tfm_train, tfm_val = make_transforms(img_size)
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    test_dir  = os.path.join(root, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=tfm_train)
    val_ds   = datasets.ImageFolder(val_dir,   transform=tfm_val)
    test_ds  = datasets.ImageFolder(test_dir,  transform=tfm_val) if os.path.isdir(test_dir) else None

    if use_combined:
        comb_train_tf = datasets.ImageFolder(train_dir, transform=tfm_train)
        comb_val_tf   = datasets.ImageFolder(val_dir,   transform=tfm_train)
        combined = ConcatDataset([comb_train_tf, comb_val_tf])
        return combined, val_ds, test_ds, train_ds.classes

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = None
    if test_ds:
        test_dl = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    return (train_dl, val_dl, test_dl, train_ds.classes)

def class_weights_from_ds(imagefolder, device):
    # expects ImageFolder-like object with .targets
    counts = np.bincount(imagefolder.targets)
    num_classes = len(counts)
    weights = (1.0 / counts) * (counts.sum() / num_classes)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return (correct / total), (loss_sum / total), np.array(y_true), np.array(y_pred)

def roc_auc_if_binary(model, loader, device):
    # returns (auc, fpr, tpr) if binary, else None
    model.eval()
    y_true, y_proba1 = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prob = torch.softmax(model(x), dim=1)
            if prob.shape[1] != 2:
                return None
            y_true.extend(y.cpu().numpy().tolist())
            y_proba1.extend(prob[:, 1].cpu().numpy().tolist())
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_proba1)
    return float(auc(fpr, tpr)), fpr.tolist(), tpr.tolist()

def train_standard(task, data_root, out_dir, epochs, batch_size, img_size, device, lr_head=5e-4, lr_body=1e-6, weight_decay=1e-4):
    # standard train/val (used for `condition`, and as fallback for `pressure`)
    train_dl, val_dl, test_dl, classes = make_loaders(data_root, batch_size, img_size)
    model = build_model(num_classes=len(classes), pretrained=True).to(device)

    # optional class weights (useful for condition if imbalanced)
    # derive from the training ImageFolder inside the DataLoader
    # we create a temporary ImageFolder with same transform to fetch .targets
    train_ds_for_weights = train_dl.dataset
    loss_weights = class_weights_from_ds(train_ds_for_weights.dataset if hasattr(train_ds_for_weights, 'dataset') else train_ds_for_weights, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)

    opt = torch.optim.AdamW(
        [{"params": model.layer4.parameters(), "lr": lr_body},
         {"params": model.fc.parameters(),    "lr": lr_head}],
        weight_decay=weight_decay
    )

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)

        train_loss = run_loss / len(train_dl.dataset)
        val_acc, val_loss, y_true, y_pred = evaluate(model, val_dl, loss_fn, device)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=ep)
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": classes}, os.path.join(out_dir, "model.pt"))
            with open(os.path.join(out_dir, "labels.json"), "w") as f:
                json.dump(classes, f)

    # test (optional)
    if test_dl:
        test_acc, test_loss, yt, yp = evaluate(model, test_dl, loss_fn, device)
        mlflow.log_metrics({"test_acc": test_acc, "test_loss": test_loss})
        if len(set(yt.tolist())) == 2:
            roc = roc_auc_if_binary(model, test_dl, device)
            if roc:
                mlflow.log_metric("test_auc", roc[0])

def train_pressure_kfold(data_root, out_dir, epochs, batch_size, img_size, k_folds, device, lr=1e-3, weight_decay=1e-4, patience=4):
    # combine train+val for k-fold, keep test separate
    combined, val_ds_for_tf, test_ds, classes = make_loaders(data_root, batch_size, img_size, use_combined=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True) if test_ds else None

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_scores = []
    best_model_state = None
    best_cv = 0.0

    for fold, (tr_idx, va_idx) in enumerate(kf.split(combined), start=1):
        # fold datasets
        train_subset = Subset(combined, tr_idx)
        val_subset   = Subset(combined, va_idx)

        # apply transforms: training (already set) vs validation (need val transforms)
        _, tfm_val = make_transforms(img_size)
        # build a mirror combined dataset but with val transforms for proper eval
        train_dir = os.path.join(data_root, "train")
        val_dir   = os.path.join(data_root, "val")
        val_mirror = ConcatDataset([
            datasets.ImageFolder(train_dir, transform=tfm_val),
            datasets.ImageFolder(val_dir,   transform=tfm_val)
        ])
        val_subset.dataset = val_mirror  # only the dataset ref is needed

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)

        model = build_model(num_classes=len(classes), pretrained=True).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

        best = 0.0; patience_ctr = 0
        with mlflow.start_run(run_name=f"pressure_fold_{fold}"):
            mlflow.log_params({"fold": fold, "epochs": epochs, "batch_size": batch_size, "lr": lr})
            for ep in range(1, epochs+1):
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

                if acc > best:
                    best = acc; patience_ctr = 0
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break

        cv_scores.append(best)
        if best > best_cv:
            best_cv = best
            best_model_state = best_state

    # final best model
    final = build_model(num_classes=len(classes), pretrained=True).to(device)
    final.load_state_dict({k: v for k, v in best_model_state.items()})
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": final.state_dict(), "classes": classes, "cv_scores": cv_scores}, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(classes, f)

    # evaluate on test if exists
    if test_dl:
        loss_fn = torch.nn.CrossEntropyLoss()
        test_acc, test_loss, yt, yp = evaluate(final, test_dl, loss_fn, device)
        mlflow.log_metrics({"test_acc": test_acc, "test_loss": test_loss})
        if len(set(yt.tolist())) == 2:
            roc = roc_auc_if_binary(final, test_dl, device)
            if roc:
                mlflow.log_metric("test_auc", roc[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["condition","pressure"], required=True)
    ap.add_argument("--data", required=True, help="root folder with train/ val/ (and test/ for pressure)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--k_folds", type=int, default=0, help="for pressure: 0 means no k-fold; >1 enables k-fold CV")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlflow.set_experiment(f"Tyre_Model_{args.task}")

    with mlflow.start_run():
        mlflow.log_params({
            "task": args.task,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "k_folds": args.k_folds
        })
        if args.task == "pressure" and args.k_folds and args.k_folds > 1:
            train_pressure_kfold(
                data_root=args.data, out_dir=args.out, epochs=args.epochs,
                batch_size=args.batch_size, img_size=args.img_size,
                k_folds=args.k_folds, device=device
            )
        else:
            train_standard(
                task=args.task, data_root=args.data, out_dir=args.out,
                epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size,
                device=device
            )
        print("✅ Training completed. Artifacts in:", args.out)

if __name__ == "__main__":
    main()
