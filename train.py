# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCIFARConvNet


def get_dataloaders(
    batch_size: int = 64,
    num_train_samples: int = 500,
    num_test_samples: int = 200,
    num_workers: int = 2,
):
    """
    Create tiny synthetic image dataset using FakeData.
    No downloads, super fast. Perfect for Jenkins CI.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    num_classes = 10

    train_dataset = datasets.FakeData(
        size=num_train_samples,
        image_size=(3, 32, 32),
        num_classes=num_classes,
        transform=transform,
    )

    test_dataset = datasets.FakeData(
        size=num_test_samples,
        image_size=(3, 32, 32),
        num_classes=num_classes,
        transform=transform,
    )

    class_names = [f"class_{i}" for i in range(num_classes)]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, class_names


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)  # CI: 1 epoch
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--num_train_samples", type=int, default=500)
    parser.add_argument("--num_test_samples", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = get_dataloaders(
        batch_size=args.batch_size,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        num_workers=args.num_workers,
    )

    model = SimpleCIFARConvNet(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
            f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

    # Save model
    model_path = os.path.join(args.output_dir, "fake_cifar_cnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save class names
    class_names_path = os.path.join(args.output_dir, "class_names.txt")
    with open(class_names_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Saved class names to {class_names_path}")


if __name__ == "__main__":
    main()
