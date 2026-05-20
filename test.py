# test.py

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF

from train import PetCNN


def make_grey_image_and_mask(image, trimap, background_value=128):
    image = image.convert("RGB")

    image_array = np.array(image).copy()
    trimap_array = np.array(trimap)

    pet_mask = trimap_array != 2
    background_mask = trimap_array == 2

    image_array[background_mask] = background_value

    mask_array = np.zeros(trimap_array.shape, dtype=np.uint8)
    mask_array[pet_mask] = 255

    grey_image = Image.fromarray(image_array)
    mask_image = Image.fromarray(mask_array)

    return grey_image, mask_image


class PetTrimapDataset(Dataset):
    def __init__(self, root, split, download=True):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types=("category", "segmentation"),
            download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        label, trimap = target

        image, mask = make_grey_image_and_mask(image, trimap)

        image = TF.resize(image, (224, 224))
        mask = TF.resize(
            mask,
            (224, 224),
            interpolation=TF.InterpolationMode.NEAREST
        )

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        mask_tensor = TF.to_tensor(mask)

        input_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        return input_tensor, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = PetTrimapDataset(
        root="data",
        split="test",
        download=True
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
