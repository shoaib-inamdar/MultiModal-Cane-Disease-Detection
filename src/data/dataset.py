import random
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class SugarcaneDataset(Dataset):
    CLASS_NAMES =["Healthy", "Red Rot", "Grassy Shoot", "Smut"]
    NUM_CLASSES=4
    def __init__(self, config, split="train", transform=None,
           synthetic=True, synthetic_metadata=True,
           images_dir=None, metadata_csv=None):
           assert split in ["train", "val", "test"], (
            f"Invalid split: {split}. "
            "Split must be one of: train, val, test"
        )
           self.config = config,
           self.split = split,
           self.transform = transform,
           self.synthetic = synthetic,
           self.synthetic_metadata = synthetic_metadata

           image_size =config['data']['image_size']
           num_classes =config['data']['num_classes']
           if synthetic:
            self._setup_synthetic(config)
           else:
            assert images_dir is not None, (
                "images_dir required when synthetic=False"
            )
            self._setup_real(images_dir, metadata_csv)
    def _setup_synthetic(self,config):
        total = config["data"]["num_synthetic_samples"]

        n_train= int(total*0.8)
        n_val=int(total* 0.1)
        n_test=total- n_train - n_val

        if self.split=="train":
            self.length=n_train
        elif self.split=="val":
            self.length=n_val
        else:
            self.length=n_test
        all_labels = np.random.RandomState(42).randint( 0,self.num_classes,total)

        if self.split=="train":
            self.labels=all_labels[:n_train]
        elif self.split=="val":
            self.labels=all_labels[n_train:n_train + n_val]
        else:
            self.labels=all_labels[n_train+n_val:]

    def _setup_real(self, images_dir, metadata_csv):
        raise NotImplementedError(
            "Real dataset loading not yet implemented. "
            "Use synthetic=True for Stage A."
        )
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        if self.synthetic:
            return self._get_synthetic_sample(idx)

        return self._get_real_sample(idx)

    def _get_synthetic_sample(self, idx: int) -> dict:
        image = torch.rand(
            3,
            self.image_size,
            self.image_size,
        )

        metadata = self._generate_synthetic_metadata()

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )
        return {
            "image": image,
            "metadata": metadata,
            "label": label,
        }
    def _generate_synthetic_metadata(self) -> torch.FloatTensor:
        temperature = random.uniform(15.0, 45.0) / 50.0
        humidity = random.uniform(30.0, 100.0) / 100.0
        soil_moisture = random.uniform(10.0, 90.0) / 100.0
        rainfall = random.uniform(0.0, 200.0) / 300.0

        return torch.tensor(
            [
                temperature,
                humidity,
                soil_moisture,
                rainfall,
            ],
            dtype=torch.float32,
        )
        
    def _get_real_sample(self, idx: int) -> dict:
        raise NotImplementedError("Stage B/C not yet implemented.")


