# train.py

import numpy as np
from PIL import Image, ImageOps

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF


IMAGE_SIZE = 224
PADDING = 25
BACKGROUND_VALUE = 128


class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


class PetCNN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            BasicBlock(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            BasicBlock(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            BasicBlock(128),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_pet_crop_and_mask(image, trimap):
    image = image.convert("RGB")

    image_array = np.array(image).copy()
    trimap_array = np.array(trimap)

    pet_mask = trimap_array != 2
    background_mask = trimap_array == 2

    image_array[background_mask] = BACKGROUND_VALUE

    mask_array = np.zeros(trimap_array.shape, dtype=np.uint8)
    mask_array[pet_mask] = 255

    grey_image = Image.fromarray(image_array)
    mask_image = Image.fromarray(mask_array)

    box = mask_image.getbbox()

    if box is None:
        return grey_image, mask_image

    left, top, right, bottom = box

    width = right - left
    height = bottom - top

    side = max(width, height) + 2 * PADDING
    side = max(side, 1)

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    crop_left = center_x - side // 2
    crop_top = center_y - side // 2
    crop_right = crop_left + side
    crop_bottom = crop_top + side

    square_image = Image.new(
        "RGB",
        (side, side),
        (BACKGROUND_VALUE, BACKGROUND_VALUE, BACKGROUND_VALUE)
    )
    square_mask = Image.new("L", (side, side), 0)

    safe_left = max(crop_left, 0)
    safe_top = max(crop_top, 0)
    safe_right = min(crop_right, image.width)
    safe_bottom = min(crop_bottom, image.height)

    image_crop = grey_image.crop((safe_left, safe_top, safe_right, safe_bottom))
    mask_crop = mask_image.crop((safe_left, safe_top, safe_right, safe_bottom))

    paste_x = max(0, -crop_left)
    paste_y = max(0, -crop_top)

    square_image.paste(image_crop, (paste_x, paste_y))
    square_mask.paste(mask_crop, (paste_x, paste_y))

    return square_image, square_mask


class PetTrimapDataset(Dataset):
    def __init__(self, root, split, train=True, download=True):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types=("category", "segmentation"),
            download=download
        )

        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        label, trimap = target

        image, mask = make_pet_crop_and_mask(image, trimap)

        if self.train and np.random.rand() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        image = TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = TF.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
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

    train_dataset = PetTrimapDataset(
        root="data",
        split="trainval",
        train=True,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    model = PetCNN(num_classes=37).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()

        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}: accuracy = {accuracy:.2f}%")

        torch.save(model.state_dict(), f"model_epoch_{epoch + 1:02d}.pth")
        torch.save(model.state_dict(), "model.pth")
        print(f"Saved checkpoint for epoch {epoch + 1}")


if __name__ == "__main__":
    main()
