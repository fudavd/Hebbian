import copy
import os
import time
import logging
from datetime import datetime
import json
import pandas as pd

from pathlib import Path
import re
from typing import List

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    file_list.sort(key=natural_keys)
    return file_list


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def select_class_set(dataset, class_ids: List[int]):
    """
    Create sub-task data set

    :param dataset: full data set
    :param class_ids: List of sub-task class ids
    :return: dataset with sub-task data
    """
    idx = dataset.targets == np.nan
    for ids in class_ids:
        idx = idx + (dataset.targets == ids)
    dataset_temp = copy.deepcopy(dataset)
    dataset_temp.targets = dataset.targets[idx]
    dataset_temp.data = dataset.data[idx]
    return dataset_temp


class AugmentWithRotations(torch.utils.data.Dataset):
    def __init__(self, base_train, base_test, rotations, transform):
        self.data = []
        self.targets = []

        all_datasets = [base_train, base_test]
        offset = 0
        for dset in all_datasets:
            for rot_idx, rotation in enumerate(rotations):
                for i in range(len(dset)):
                    img, label = dset[i]

                    img = transforms.F.rotate(img, rotation)
                    img = transform(img)
                    self.data.append(img)

                    new_label = label * len(rotations) + rot_idx + offset
                    self.targets.append(new_label)
            offset += len(self.targets)

        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def LoadDataFolder(path='./data', transform=None):
    dset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, dset):
            self.data = np.array([p for p, _ in dset.samples])
            self.targets = torch.tensor(dset.targets)
            self.samples = dset.samples
            self.classes = dset.classes
            self.class_to_idx = dset.class_to_idx
            self.loader = dset.loader
            self.transform = dset.transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img = self.loader(self.data[idx])
            if self.transform:
                img = self.transform(img)
            return img, self.targets[idx]

    return WrappedDataset(dset)


dataset_map = {
    'MNIST': torchvision.datasets.MNIST,
    'Omniglot': torchvision.datasets.Omniglot,
    'MiniImageNet': None,  # Placeholder, handled separately below
    'CUB200': None,
}
def get_dataset(name: str):

    # Validate input
    if name not in dataset_map:
        supported = list(dataset_map.keys())
        raise ValueError(f"Dataset '{name}' not supported. Choose from: {supported}")

    # === Normalization values per dataset ===
    norm_values = {
        'MNIST': ((0.1307,), (0.3081,)),
        'Omniglot': ((0.5,), (0.5,)),
        'MiniImageNet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'CUB200': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }

    mean, std = norm_values[name]
    transform_list = [
        transforms.Resize((84, 84)) if name in ['MiniImageNet', 'CUB200'] else transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform_list)

    # === Load dataset ===
    if name == 'Omniglot':
        base_train = torchvision.datasets.Omniglot(root='./data', background=True, download=True)
        base_test = torchvision.datasets.Omniglot(root='./data', background=False, download=True)

        rotations = [0, 90, 180, 270]
        rotated_dataset = AugmentWithRotations(base_train, base_test, rotations, transform)
        unique_classes = np.unique(rotated_dataset.targets)
        unique_class_ids = np.random.permutation(np.arange(len(unique_classes)))
        train_dataset = select_class_set(rotated_dataset, unique_class_ids[:1200])
        test_dataset = select_class_set(rotated_dataset, unique_class_ids[1200:1623])
    # elif name == 'TieredImagenet':
    elif name == 'MiniImageNet':
        # MiniImageNet is not included in torchvision.
        # Expect data in ./data/mini-imagenet/train/ and ./data/mini-imagenet/val/
        train_dataset = LoadDataFolder(path='./data/mini-imagenet/train', transform=transform)
        test_dataset = LoadDataFolder(path='./data/mini-imagenet/val', transform=transform)
    elif name == 'CUB200':
        train_dataset = LoadDataFolder(path='./data/CUB_200/train', transform=transform)
        test_dataset = LoadDataFolder(path='./data/CUB_200/test', transform=transform)
    else:
        dataset_class = dataset_map[name]
        train_dataset = dataset_class(root='./data', train=True, transform=transform, download=True)
        test_dataset = dataset_class(root='./data', train=False, transform=transform, download=True)

    # === Ensure .targets and .classes exist ===
    for dset in [train_dataset, test_dataset]:
        if not hasattr(dset, "targets"):
            if hasattr(dset, "imgs"):
                dset.targets = [label for _, label in dset.imgs]
            else:
                dset.targets = np.array([y for _, y in dset])
        if not hasattr(dset, "classes"):
            if hasattr(dset, "class_to_idx"):
                dset.classes = list(dset.class_to_idx.keys())
            else:
                dset.classes = sorted(np.unique(dset.targets))

    return train_dataset, test_dataset


def split_dataset_by_class(dataset, n_meta_train_classes=5, n_meta_val_classes=5):
    """
    Split a dataset (like MNIST) into meta-train, and meta-val subsets by class.
    """
    labels = np.array(dataset.targets)
    all_classes = np.unique(labels)

    np.random.shuffle(all_classes)

    meta_train_classes = all_classes[:n_meta_train_classes]
    meta_val_classes = all_classes[-n_meta_val_classes:]

    return (
        select_class_set(dataset, meta_train_classes.tolist()),
        select_class_set(dataset, meta_val_classes.tolist()),
    )

class TrainingLogger:
    def __init__(self, log_dir="./logs", experiment_name="meta_learning"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = time.time()

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Training history
        self.history = {
            'meta_epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'grad_norms': [],
            'timestamp': []
        }

    def setup_logging(self):
        """Setup file and console logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}_{timestamp}.log")

        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Log file: {log_file}")

    def log_epoch(self, epoch, loss, accuracy, val_loss, val_acc, optimizer, model):
        """Log epoch-level metrics and model info"""
        current_lr = optimizer.param_groups[0]['lr']

        # Calculate gradient norms
        grad_norms = {}
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                total_norm += grad_norm ** 2
            else:
                grad_norms[name] = 0.0

        total_norm = total_norm ** 0.5

        # Update history
        self.history['meta_epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(current_lr)
        self.history['grad_norms'].append(grad_norms)
        self.history['timestamp'].append(time.time())

        # Log metrics
        self.logger.info(
            f"Epoch {epoch:04d} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}% | "
            f"LR: {current_lr:.2e} | "
            f"Grad Norm: {total_norm:.4f}"
        )

        # Log detailed gradient information every 10 epochs
        if epoch % 10 == 0:
            self.logger.debug("Detailed gradient norms:")
            for name, norm in grad_norms.items():
                self.logger.debug(f"  {name}: {norm:.6f}")

    def log_meta_batch(self, meta_batch, loss, accuracy):
        """Log meta-batch level metrics (for progress bar)"""
        logging.debug(f"Meta-batch {meta_batch}: loss={loss:.4f}, accuracy={accuracy * 100:.2f}%")

    def save_checkpoint(self, model, optimizer, epoch, filename=None):
        """Save training checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = os.path.join(self.log_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': self.history['loss'][-1],
            'accuracy': self.history['accuracy'][-1],
            'val_loss': self.history['val_loss'][-1],
            'val_acc': self.history['val_acc'][-1],
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_training_history(self):
        """Save training history to CSV"""
        csv_path = os.path.join(self.log_dir, "training_history.csv")

        # Create DataFrame from history
        df_data = {
            'meta_epoch': self.history['meta_epoch'],
            'loss': self.history['loss'],
            'accuracy': self.history['accuracy'],
            'val_loss': self.history['val_loss'],
            'val_acc': self.history['val_acc'],
            'learning_rate': self.history['learning_rate'],
            'timestamp': self.history['timestamp']
        }

        # Add gradient norms as separate columns
        for epoch_idx, grad_norms in enumerate(self.history['grad_norms']):
            for param_name, norm in grad_norms.items():
                col_name = f"grad_{param_name.replace('.', '_')}"
                if col_name not in df_data:
                    df_data[col_name] = [0.0] * len(self.history['meta_epoch'])
                df_data[col_name][epoch_idx] = norm

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Training history saved: {csv_path}")

        return df

    def log_experiment_config(self, config):
        """Log experiment configuration"""
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        # Save config to JSON
        config_path = os.path.join(self.log_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def get_training_time(self):
        """Get elapsed training time"""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
