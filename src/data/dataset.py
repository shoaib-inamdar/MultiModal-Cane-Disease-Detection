import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FOLDER_TO_IDX: dict[str, int] = {
    "Healthy Leaves": 0,
    "Grassy shoot": 1,
    "smut": 2,
    "Pokkah Boeng": 3,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


class SugarcaneDataset(Dataset):
    CLASS_NAMES = ["Healthy", "Grassy Shoot", "Smut", "Pokkah Boeng"]
    NUM_CLASSES = 4

    def __init__(
        self,
        config,
        split="train",
        transform=None,
        synthetic=True,
        synthetic_metadata=True,
        images_dir=None,
        metadata_csv=None,
    ):
        assert split in ["train", "val", "test"], (
            f"Invalid split: {split}. Split must be one of: train, val, test"
        )
        self.config = config
        self.split = split
        self.transform = transform
        self.synthetic = synthetic
        self.synthetic_metadata = synthetic_metadata

        self.image_size = config["data"]["image_size"]
        self.num_classes = config["data"]["num_classes"]

        if synthetic:
            self._setup_synthetic(config)
        else:
            assert images_dir is not None, "images_dir required when synthetic=False"
            self._setup_real(images_dir, metadata_csv)

    def _setup_synthetic(self, config):
        total = config["data"]["num_synthetic_samples"]

        n_train = int(total * 0.8)
        n_val = int(total * 0.1)
        n_test = total - n_train - n_val

        if self.split == "train":
            self.length = n_train
        elif self.split == "val":
            self.length = n_val
        else:
            self.length = n_test

        all_labels = np.random.RandomState(42).randint(0, self.num_classes, total)

        if self.split == "train":
            self.labels = all_labels[:n_train]
        elif self.split == "val":
            self.labels = all_labels[n_train : n_train + n_val]
        else:
            self.labels = all_labels[n_train + n_val :]


    def _setup_real(self, images_dir: str, metadata_csv):
        """
        Walk images_dir and collect image paths + integer labels.

        Expected folder structure (one folder per class):
            images_dir/
                Healthy Leaves/   (images)
                Grassy shoot/     (images)
                smut/             (images)
                Pokkah Boeng/     (images)
                ...               (other folders are ignored)

        Splits: 80% train / 10% val / 10% test (reproducible, seed=42).
        Metadata: remains synthetic for Stage B (no paired sensor CSV yet).
        When Stage C real sensor CSVs are available, pass metadata_csv
        and replace _generate_synthetic_metadata() with a CSV lookup.
        """
        root = Path(images_dir)
        all_paths: list[Path] = []
        all_labels: list[int] = []

        for folder_name, class_idx in FOLDER_TO_IDX.items():
            class_dir = root / folder_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix in IMAGE_EXTENSIONS:
                    all_paths.append(img_path)
                    all_labels.append(class_idx)

        if len(all_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {images_dir}. "
                f"Expected sub-folders: {list(FOLDER_TO_IDX.keys())}"
            )


        rng = np.random.RandomState(42)
        indices = rng.permutation(len(all_paths))

        n_train = int(len(indices) * 0.8)
        n_val = int(len(indices) * 0.1)

        if self.split == "train":
            split_idx = indices[:n_train]
        elif self.split == "val":
            split_idx = indices[n_train : n_train + n_val]
        else:
            split_idx = indices[n_train + n_val :]

        self.image_paths = [all_paths[i] for i in split_idx]
        self.labels = [all_labels[i] for i in split_idx]
        self.length = len(self.image_paths)

        # ImageNet normalisation (same stats the Swin backbone was pretrained on)
        self._real_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        if self.synthetic:
            return self._get_synthetic_sample(idx)
        return self._get_real_sample(idx)

    def _get_synthetic_sample(self, idx: int) -> dict:
        image = torch.rand(3, self.image_size, self.image_size)
        metadata = self._generate_synthetic_metadata()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"image": image, "metadata": metadata, "label": label}

    def _get_real_sample(self, idx: int) -> dict:
        """
        Load a real image from disk.
        Metadata is synthetic for Stage B (no paired sensor CSV yet).
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        image = self._real_transform(img)
        metadata = self._generate_synthetic_metadata()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"image": image, "metadata": metadata, "label": label}

    def _generate_synthetic_metadata(self) -> torch.Tensor:
        temperature = random.uniform(15.0, 45.0) / 50.0
        humidity = random.uniform(30.0, 100.0) / 100.0
        soil_moisture = random.uniform(10.0, 90.0) / 100.0
        rainfall = random.uniform(0.0, 200.0) / 300.0

        return torch.tensor(
            [temperature, humidity, soil_moisture, rainfall],
            dtype=torch.float32,
        )
