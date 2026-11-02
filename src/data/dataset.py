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

IDX_TO_FOLDER: dict[int, str] = {
    idx: folder_name for folder_name, idx in FOLDER_TO_IDX.items()
}

# Stage B approximate metadata by disease class
# Based on known climate triggers (IMD Maharashtra data)

DISEASE_CLIMATE = {
    "Healthy Leaves": {
        "temperature_range": (22, 28),  # °C -- dry cool season
        "humidity_range": (50, 70),  # %
        "rainfall_range": (3, 50),  # mm
        "soil_moisture": (40, 60),  # %
    },
    "Grassy shoot": {
        "temperature_range": (24, 32),  # °C -- warm
        "humidity_range": (70, 90),  # % -- high
        "rainfall_range": (100, 300),  # mm -- monsoon
        "soil_moisture": (60, 80),  # %
    },
    "smut": {
        "temperature_range": (26, 34),  # °C -- warm dry
        "humidity_range": (55, 75),  # % -- moderate
        "rainfall_range": (10, 80),  # mm -- drought stress
        "soil_moisture": (30, 55),  # %
    },
    "Pokkah Boeng": {
        "temperature_range": (25, 33),  # °C -- warm
        "humidity_range": (65, 85),  # % -- post-rain
        "rainfall_range": (50, 200),  # mm
        "soil_moisture": (50, 75),  # %
    },
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

        Note: rglob() is used instead of iterdir() so images are found
        regardless of nesting depth. This handles the common case where
        each class zip extracts into a same-named subfolder, e.g.:
            images_dir/Grassy shoot/Grassy shoot/image001.jpg
        """
        root = Path(images_dir)
        all_paths: list[Path] = []
        all_labels: list[int] = []

        for folder_name, class_idx in FOLDER_TO_IDX.items():
            class_dir = root / folder_name
            if not class_dir.exists():
                continue
            # rglob finds images at any depth inside the class folder
            found = sorted(
                p for p in class_dir.rglob("*") if p.suffix in IMAGE_EXTENSIONS
            )
            all_paths.extend(found)
            all_labels.extend([class_idx] * len(found))

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

        If self.transform is set (e.g. augmentation pipeline passed at init),
        it is applied instead of the default _real_transform.
        Metadata is synthetic for Stage B (no paired sensor CSV yet).
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(img)
        else:
            image = self._real_transform(img)

        class_idx = self.labels[idx]

        metadata = self._get_approximate_metadata(class_idx)

        label = torch.tensor(class_idx, dtype=torch.long)

        return {
            "image": image,
            "metadata": metadata,
            "label": label,
        }

    def _get_approximate_metadata(self, class_idx: int) -> torch.Tensor:
        """
        Generate approximate metadata using known disease-climate correlations.
        Better than pure random -- gives the model a real learning signal.
        """
        class_name = IDX_TO_FOLDER[class_idx]  # reverse mapping
        climate = DISEASE_CLIMATE.get(
            class_name,
            {
                "temperature_range": (20, 35),
                "humidity_range": (50, 90),
                "rainfall_range": (0, 300),
                "soil_moisture": (30, 80),
            },
        )

        t_lo, t_hi = climate["temperature_range"]
        h_lo, h_hi = climate["humidity_range"]
        r_lo, r_hi = climate["rainfall_range"]
        s_lo, s_hi = climate["soil_moisture"]

        # Sample within disease-specific range (adds realism vs pure random)
        temperature = random.uniform(t_lo, t_hi)
        humidity = random.uniform(h_lo, h_hi)
        rainfall = random.uniform(r_lo, r_hi)
        soil_moisture = random.uniform(s_lo, s_hi)

        # Normalise to [0,1] -- same ranges as training
        return torch.tensor(
            [
                (temperature - 15) / 30,
                (humidity - 20) / 80,
                soil_moisture / 100,
                rainfall / 300,
            ],
            dtype=torch.float32,
        )

    def _generate_synthetic_metadata(self) -> torch.Tensor:
        temperature = random.uniform(15.0, 45.0) / 50.0
        humidity = random.uniform(30.0, 100.0) / 100.0
        soil_moisture = random.uniform(10.0, 90.0) / 100.0
        rainfall = random.uniform(0.0, 200.0) / 300.0

        return torch.tensor(
            [temperature, humidity, soil_moisture, rainfall],
            dtype=torch.float32,
        )
