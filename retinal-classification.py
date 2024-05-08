import os
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(
    description="Train a Convolutional Autoencoder with classification"
)
parser.add_argument(
    "--lambda_param",
    type=float,
    default=0.5,
    help="Lambda to balance reconstruction and classification losses",
)
args = parser.parse_args()


# Define the model
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


# Transformation and data loading
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # Normalize the datasets
    ]
)

trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Model setup
model = ConvAutoencoder()
criterion_recon = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare saving results
results_folder = "./training_results"
os.makedirs(results_folder, exist_ok=True)
result_file_path = os.path.join(
    results_folder, f"results_lambda_{args.lambda_param}.txt"
)

# Training loop
history = {"reconstruction": [], "classification": []}
for epoch in range(10):
    recon_losses = []
    class_losses = []
    for data in trainloader:
        inputs, classes = data
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
        f'Epoch {epoch+1}, Reconstruction Loss: {history["reconstruction"][-1]}, Classification Loss: {history["classification"][-1]}'
    )

# Save results to file
with open(result_file_path, "w") as f:
    f.write(f"Lambda: {args.lambda_param}\n")
    f.write("Epoch,Reconstruction Loss,Classification Loss\n")
    for epoch in range(10):
        f.write(
            f"{epoch+1},{history['reconstruction'][epoch]},{history['classification'][epoch]}\n"
        )

print(f"Results saved to {result_file_path}")
