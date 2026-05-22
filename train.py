# train.py

import random
import numpy as np
from PIL import Image, ImageOps

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF
from tqdm import tqdm


IMAGE_SIZE = 224
PADDING = 25
BACKGROUND_VALUE = 128
NUM_CLASSES = 37


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class PetCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256)
        )

        self.layer5 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
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

        if self.train and random.random() < 0.5:
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
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    epochs = 30
    learning_rate = 8e-4

    train_dataset = PetTrimapDataset(
        root="data",
        split="trainval",
        train=True,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = PetCNN(num_classes=NUM_CLASSES).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=3e-5
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )

    for epoch in range(epochs):
        model.train()

        correct = 0
        total = 0
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.2f}%"
            )

        accuracy = 100 * correct / total
        average_loss = total_loss / total

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.8f} | "
            f"Train Loss: {average_loss:.4f} | "
            f"Train Accuracy: {accuracy:.2f}%"
        )

        checkpoint_name = f"roi_epoch_{epoch + 1:02d}.pth"
        torch.save(model.state_dict(), checkpoint_name)
        torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
