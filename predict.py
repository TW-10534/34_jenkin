# predict.py
import argparse
import torch
from torchvision import transforms
from PIL import Image
import os

from model import SimpleCIFARConvNet

def load_class_names(path: str):
    with open(path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def get_transform():
    # same normalization as test set
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="artifacts/cifar_cnn.pt")
    parser.add_argument("--classes_path", type=str, default="artifacts/class_names.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.classes_path):
        raise FileNotFoundError(f"Classes file not found: {args.classes_path}")

    class_names = load_class_names(args.classes_path)

    model = SimpleCIFARConvNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transform()

    img = Image.open(args.image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # [1, 3, 32, 32]

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_class = class_names[pred_idx]
        pred_prob = probs[0, pred_idx].item()

    print(f"Predicted class: {pred_class} (confidence: {pred_prob:.4f})")


if __name__ == "__main__":
    main()
