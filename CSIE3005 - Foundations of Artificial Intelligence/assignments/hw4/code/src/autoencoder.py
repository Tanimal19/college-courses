import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

"""
Implementation of Autoencoder
"""


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, X, epochs=10, batch_size=32):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters())

        self.train()  # Set the model to training mode
        loss_history = []

        for epoch in tqdm(range(epochs)):
            # In this case, batch_size is the same as dataset size, so one batch per epoch
            outputs = self(X_tensor)
            loss = criterion(outputs, X_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            # Optional: Print loss every few epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        return loss_history

    def transform(self, X):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            encoded = self.encoder(X_tensor)
            return encoded.numpy()

    def reconstruct(self, X):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            elif isinstance(X, torch.Tensor):
                X_tensor = X.clone().detach()
            else:
                raise TypeError("Input must be a NumPy array or a PyTorch tensor")

            decoded = self.decoder(X_tensor)
            return decoded.numpy()


"""
Implementation of DenoisingAutoencoder
"""


class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2, arch=None):
        super(Autoencoder, self).__init__()  # Call parent's __init__ explicitly

        self.noise_factor = noise_factor

        # Architecture 1: Shallower and Thinner
        if arch == "shallow":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, input_dim),
            )

        # Architecture 2: Deeper and Fatter
        elif arch == "deep":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim * 2),
                nn.ReLU(),
                nn.Linear(encoding_dim * 2, encoding_dim),
                nn.ReLU(),
                nn.Linear(encoding_dim, encoding_dim // 2),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim // 2, encoding_dim),
                nn.ReLU(),
                nn.Linear(encoding_dim, encoding_dim * 2),
                nn.ReLU(),
                nn.Linear(encoding_dim * 2, input_dim),
            )

        # Original Architecture
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim),
                nn.Linear(encoding_dim, encoding_dim // 2),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim // 2, encoding_dim),
                nn.Linear(encoding_dim, input_dim),
            )

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = x + noise
        return noisy_x

    def fit(self, X, epochs=10, batch_size=32, optimizer_name="adam", lr=0.001):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        criterion = nn.MSELoss()

        if optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer_name == "adagrad":
            optimizer = optim.Adagrad(self.parameters(), lr=lr)
        elif optimizer_name == "adadelta":
            optimizer = optim.Adadelta(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

        self.train()  # Set the model to training mode
        loss_history = []

        for epoch in tqdm(range(epochs)):
            # In this case, batch_size is the same as dataset size, so one batch per epoch
            noisy_X_tensor = self.add_noise(X_tensor)
            outputs = self(noisy_X_tensor)  # Pass noisy input through the autoencoder
            loss = criterion(outputs, X_tensor)  # Compare output to original input

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            # Optional: Print loss every few epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        return loss_history
