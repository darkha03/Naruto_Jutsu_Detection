from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small on class-folder image dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_DIR / "images" / "cropped_by_label",
        help="Root folder containing one subfolder per class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "train_result" / "mobilenetv3_small",
        help="Where to store checkpoints and training metadata.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for train/val loaders."
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="Input image size (square)."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio from the full dataset.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test split ratio from the full dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="DataLoader worker processes."
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include 'background' class if present in dataset folder.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick experiments/debug runs.",
    )
    parser.add_argument(
        "--use-balanced-sampler",
        action="store_true",
        help="Use weighted random sampler during training.",
    )
    return parser.parse_args()


class FilteredImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        include_background: bool,
        transform,
    ) -> None:
        base = datasets.ImageFolder(root=root, transform=None)
        selected_classes = []
        for class_name in base.classes:
            if class_name == "background" and not include_background:
                continue
            selected_classes.append(class_name)

        if not selected_classes:
            raise RuntimeError("No classes available after applying class filters.")

        self.classes = selected_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.transform = transform

        samples: list[tuple[str, int]] = []
        for path, old_idx in base.samples:
            class_name = base.classes[old_idx]
            if class_name not in self.class_to_idx:
                continue
            samples.append((path, self.class_to_idx[class_name]))

        if not samples:
            raise RuntimeError("No images found after applying class filters.")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = default_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc += accuracy(logits, labels)
            total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0
    return total_loss / total_batches, total_acc / total_batches


def build_splits(
    total_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_samples: int | None,
) -> tuple[list[int], list[int], list[int]]:
    if total_size <= 0:
        return [], [], []

    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if max_samples is not None:
        indices = indices[: max(1, min(len(indices), max_samples))]

    n = len(indices)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    if n_test + n_val >= n:
        n_test = min(n // 5, n_test)
        n_val = min(n // 5, n_val)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    if not train_idx:
        train_idx = indices
        val_idx = []
        test_idx = []

    return train_idx, val_idx, test_idx


def make_train_sampler(dataset: FilteredImageDataset, train_indices: list[int]) -> WeightedRandomSampler:
    targets = [dataset.samples[i][1] for i in train_indices]
    class_counts = Counter(targets)
    sample_weights = [1.0 / class_counts[t] for t in targets]
    weight_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weight_tensor, num_samples=len(weight_tensor), replacement=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.img_size <= 0:
        raise ValueError("--img-size must be > 0")
    if args.val_ratio < 0 or args.test_ratio < 0 or (args.val_ratio + args.test_ratio) >= 1:
        raise ValueError("Require 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    ds_for_split = FilteredImageDataset(
        root=str(data_dir),
        include_background=args.include_background,
        transform=None,
    )

    train_idx, val_idx, test_idx = build_splits(
        total_size=len(ds_for_split),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    train_ds = FilteredImageDataset(
        root=str(data_dir),
        include_background=args.include_background,
        transform=train_tf,
    )
    eval_ds = FilteredImageDataset(
        root=str(data_dir),
        include_background=args.include_background,
        transform=eval_tf,
    )

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(eval_ds, val_idx)
    test_subset = Subset(eval_ds, test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(train_ds.classes)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if not args.no_pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.to(device)

    train_labels = [train_ds.samples[i][1] for i in train_idx]
    class_counts = Counter(train_labels)

    class_weights = torch.ones(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        if class_counts.get(c, 0) > 0:
            class_weights[c] = 1.0 / class_counts[c]
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    if args.use_balanced_sampler:
        sampler = make_train_sampler(train_ds, train_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = output_dir / f"mobilenetv3_small_best_{timestamp}.pt"
    last_path = output_dir / f"mobilenetv3_small_last_{timestamp}.pt"
    labels_path = output_dir / f"mobilenetv3_small_labels_{timestamp}.json"
    report_path = output_dir / f"mobilenetv3_small_report_{timestamp}.json"

    print(f"Device: {device}")
    print(f"Dataset: {data_dir}")
    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(
        f"Split sizes -> train: {len(train_subset)}, val: {len(val_subset)}, test: {len(test_subset)}"
    )

    best_val_acc = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        batch_count = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(logits, labels)
            batch_count += 1

        train_loss = running_loss / max(1, batch_count)
        train_acc = running_acc / max(1, batch_count)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": train_ds.classes,
                    "img_size": args.img_size,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": train_ds.classes,
            "img_size": args.img_size,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "epochs": args.epochs,
        },
        last_path,
    )

    labels_payload = {
        "class_to_idx": train_ds.class_to_idx,
        "classes": train_ds.classes,
        "include_background": args.include_background,
    }
    labels_path.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    report_payload = {
        "device": str(device),
        "data_dir": str(data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "split_sizes": {
            "train": len(train_subset),
            "val": len(val_subset),
            "test": len(test_subset),
        },
        "class_counts_train": {str(k): v for k, v in class_counts.items()},
        "history": history,
        "artifacts": {
            "best_model": str(best_path),
            "last_model": str(last_path),
            "labels": str(labels_path),
        },
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Labels map: {labels_path}")
    print(f"Report: {report_path}")
    print(f"Final test metrics -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
