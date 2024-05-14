### Imports ###

import os
import argparse
import multiprocessing
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split


### CLI ###


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a Convolutional Autoencoder with classification"
    )
    parser.add_argument(
        "-f",
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "-l",
        "--lambda_param",
        type=float,
        default=0.5,
        help="Lambda to balance reconstruction and classification losses",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (cpu or cuda)",
    )

    return parser.parse_args()


### Model definition ###


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Assuming input images are scaled between 0 and 1
        )
        # Classifier
        self.classifier = nn.Linear(64 * 4 * 4, 10)  # Adjust the size accordingly

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        encoded_flat = encoded.view(encoded.size(0), -1)
        classification = self.classifier(encoded_flat)
        return decoded, classification


### Training ###


def train_fold(fold, args, device, trainset):
    # Split dataset
    num_folds = args.folds
    fold_size = len(trainset) // num_folds
    folds = random_split(
        trainset,
        [fold_size] * (num_folds - 1) + [len(trainset) - fold_size * (num_folds - 1)],
    )

    train_subsets = [x for i, x in enumerate(folds) if i != fold]
    train_subset = torch.utils.data.ConcatDataset(train_subsets)
    validation_set = folds[fold]

    trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
    validationloader = DataLoader(validation_set, batch_size=64, shuffle=False)

    # Initialize model and move it to specified device
    model = ConvAutoencoder().to(device)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop for this fold
    history = {"reconstruction": [], "classification": []}

    # Training loop
    for epoch in range(10):
        model.train()
        recon_losses, class_losses = [], []
        for data in trainloader:
            inputs, classes = data
            inputs, classes = inputs.to(device), classes.to(device)
            optimizer.zero_grad()
            outputs, predicted_classes = model(inputs)
            loss_recon = criterion_recon(outputs, inputs)
            loss_class = criterion_class(predicted_classes, classes)
            loss = loss_recon + args.lambda_param * loss_class
            loss.backward()
            optimizer.step()
            recon_losses.append(loss_recon.item())
            class_losses.append(loss_class.item())

        history["reconstruction"].append(sum(recon_losses) / len(recon_losses))
        history["classification"].append(sum(class_losses) / len(class_losses))
        print(
            f'Fold {fold+1}, Epoch {epoch+1}, Reconstruction Loss: {history["reconstruction"][-1]}, Classification Loss: {history["classification"][-1]}'
        )

    # Saving results
    results_folder = "./training_results"
    os.makedirs(results_folder, exist_ok=True)
    result_file_path = os.path.join(
        results_folder, f"results_lambda_{args.lambda_param}_fold_{fold + 1}.txt"
    )

    # Implement training and validation here
    print(f"Training and validation for fold {fold+1} completed.")


### Main ###


def main():

    # Get arguments
    args = get_args()

    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)

    # Setup multiprocessing
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    train_func = partial(train_fold, args=args, device=args.device, trainset=trainset)
    pool.map(train_func, range(args.folds))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
