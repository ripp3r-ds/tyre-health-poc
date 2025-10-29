from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Transforms ---

def make_transforms_condition(img_size: int = 224):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tfm_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return tfm_train, tfm_eval


def make_transforms_pressure(img_size: int = 224):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tfm_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=5),
        transforms.ToTensor(),
        normalize,
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return tfm_train, tfm_eval


# --- Loader ---

def load_standard_split(data_root: str,
                        batch_size: int = 32,
                        img_size: int = 224,
                        num_workers: int = 2,
                        task: str = "condition",
                        pin_memory: Optional[bool] = None) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """Create train/val/test loaders and return class names.
    Expects data_root/{train,val[,test]}/class_name/img.jpg

    pin_memory: if None, auto = True when CUDA is available; else use provided bool.
    """
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing val dir: {val_dir}")

    if task == "condition":
        tfm_train, tfm_eval = make_transforms_condition(img_size)
    else:
        tfm_train, tfm_eval = make_transforms_pressure(img_size)

    # Use ImageFolder once to infer classes
    tmp = datasets.ImageFolder(str(train_dir))
    classes = tmp.classes
    if len(classes) < 2:
        raise ValueError(f"Expected at least 2 classes; got {len(classes)} in {train_dir}")

    train_ds = datasets.ImageFolder(str(train_dir), transform=tfm_train)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=tfm_eval)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    print(f"CPU/GPU Pin memory set to: {pin_memory}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = None
    if test_dir.is_dir():
        test_ds = datasets.ImageFolder(str(test_dir), transform=tfm_eval)
        test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, classes