#!/usr/bin/env python3

import numpy as np
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
            neuralnetwork.Linear(100, 128),  # Random noise dimension to intermediate
            neuralnetwork.ReLU(),
            neuralnetwork.Linear(128, 9),  # Output the number of sensor readings
            neuralnetwork.Tanh()  # Since data is normalized between -10 and 10
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(neuralnetwork.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = neuralnetwork.Sequential(
            neuralnetwork.Linear(9, 128),
            neuralnetwork.LeakyReLU(0.2),
            neuralnetwork.Linear(128, 1),
            neuralnetwork.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def main():

    subject = 1
    exercise = 1
    unit = 2

    # Load dataset
    df = pd.read_csv(f's{subject}/e{exercise}/u{unit}/test-correct.csv', delimiter=';')

    # Drop the time index as it's not a feature
    df = df.drop(columns=['time index'])

    # Convert to pytorch tensors
    tensor_data = torch.tensor(df.values, dtype=torch.float)

    # Create a dataset and dataloader
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=32)

    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

    # Loss and optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = neuralnetwork.BCELoss()

    # Training loop
    epochs = 200
    for epoch in range(epochs):
        for i, (real_samples,) in enumerate(dataloader):
            # Training Discriminator
            real_samples_labels = torch.ones((real_samples.size(0), 1))
            latent_space_samples = torch.randn((real_samples.size(0), 100))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((real_samples.size(0), 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = criterion(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_D.step()

            # Training Generator
            latent_space_samples = torch.randn((real_samples.size(0), 100))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.ones((real_samples.size(0), 1))
            discriminator.zero_grad()
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = criterion(output_discriminator_generated, generated_samples_labels)
            loss_generator.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Loss D.: {loss_discriminator}, Loss G.: {loss_generator}")

    # Generate new data points after training
    # new_data_points = 10000
    # synthetic_data = generator(torch.randn(new_data_points, 100)).detach().numpy()

    # # Save the generated data to a file
    # output_path = f's{subject}/e{exercise}/u{unit}/synth-correct.csv'
    # np.savetxt(
    #     output_path,
    #     synthetic_data,
    #     delimiter=';',
    #     header='acc_x;acc_y;acc_z;gyr_x;gyr_y;gyr_z;mag_x;mag_y;mag_z'
    # )

    with torch.no_grad():
        z = torch.randn(10000, 100)  # Generate one sample; adjust as needed
        synthetic_data = generator(z)
        # Convert the tensor to a pandas DataFrame
        synth_df = pd.DataFrame(synthetic_data.numpy())
        synth_df.to_csv(f's{subject}/e{exercise}/u{unit}/synth-test-correct.csv', sep=';', index=False)

if __name__ == '__main__':
    main()
