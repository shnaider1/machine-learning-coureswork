# test.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import PetCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    test_dataset = datasets.OxfordIIITPet(
        root="data",
        split="test",
        target_types="category",
        download=True,
        transform=image_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    model = PetCNN(num_classes=37).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
