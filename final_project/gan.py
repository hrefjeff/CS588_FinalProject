#!/usr/bin/env python3

import pandas as pd
import torch
import torch.nn as neuralnetwork
import sklearn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class Generator(neuralnetwork.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = neuralnetwork.Sequential(
            neuralnetwork.Linear(100, 256),  # Random noise dimension to 256
            neuralnetwork.LeakyReLU(0.2),
            neuralnetwork.Linear(256, 512),
            neuralnetwork.LeakyReLU(0.2),
            neuralnetwork.Linear(512, 9),  # Output dimension matching feature size
            neuralnetwork.Tanh()  # Assuming data is scaled between -1 and 1
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(neuralnetwork.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = neuralnetwork.Sequential(
            neuralnetwork.Linear(9, 512),  # Input dimension matching feature size
            neuralnetwork.LeakyReLU(0.2),
            neuralnetwork.Linear(512, 256),
            neuralnetwork.LeakyReLU(0.2),
            neuralnetwork.Linear(256, 1),
            neuralnetwork.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def main():
    # Load dataset
    df = pd.read_csv('s1/e1/u1/template_session.txt', delimiter=';')

    # Drop the time index as it's not a feature
    df = df.drop(columns=['time index'])

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    # Convert to pytorch tensors
    tensor_data = torch.tensor(scaled_features, dtype=torch.float)

    # Create a dataset and dataloader
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # Loss function
    criterion = neuralnetwork.BCELoss()

    # Training loop
    epochs = 200  # will need to play with this value
    for epoch in range(epochs):
        for real_data, in dataloader:
            # Update discriminator with real data
            optimizer_d.zero_grad()
            real_labels = torch.ones(real_data.size(0), 1)
            real_loss = criterion(discriminator(real_data), real_labels)
            real_loss.backward()

            # Generate fake data
            z = torch.randn(real_data.size(0), 100)
            fake_data = generator(z)
            fake_labels = torch.zeros(real_data.size(0), 1)
            fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
            fake_loss.backward()
            optimizer_d.step()

            # Update generator
            optimizer_g.zero_grad()
            tricked_labels = torch.ones(real_data.size(0), 1)
            gen_loss = criterion(discriminator(fake_data), tricked_labels)
            gen_loss.backward()
            optimizer_g.step()

        print(f"Epoch {epoch+1}/{epochs}, D Loss: {real_loss+fake_loss}, G Loss: {gen_loss}")

    with torch.no_grad():
        z = torch.randn(2000, 100)  # Generate one sample; adjust as needed
        synthetic_data = generator(z)
        print(synthetic_data)

if __name__ == '__main__':
    main()
