import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pymulti as pm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os

class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size=64):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def calculate_latent_distribution(data, autoencoder, scaler):
    data_normalized = scaler.transform(data[:, :100])
    with torch.no_grad():
        latent_vectors = autoencoder.encoder(torch.tensor(data_normalized, dtype=torch.float32))
    mean = latent_vectors.mean(dim=0)
    std = latent_vectors.std(dim=0)
    return mean.numpy(), std.numpy()

def generate_latent_samples(mean, std, num_samples=5):
    samples = []
    for i in range(len(mean)):
        sample_values = np.linspace(mean[i] - 0.5*std[i], 
                                  mean[i] + 0.5*std[i], 
                                  num_samples)
        samples.append(sample_values)
    return np.array(list(itertools.product(*samples)))

def train_autoencoder(autoencoder, data, iteration,scaler, num_epochs=100, batch_size=32, lr=0.001):
    power_normalized = scaler.transform(data)
    
    tensor_data = torch.tensor(power_normalized, dtype=torch.float32)
    dataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0
        for batch in dataloader:
            inputs, _ = batch
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    # Save model
    model_path = f"./ae_models/autoencoder_{iteration}.pth"
    torch.save(autoencoder.state_dict(), model_path)
    return autoencoder

def reconstruct_and_plot(autoencoder, scaler, iteration):
    # Load sample data
    sample_data = np.load("original_data1.npy")[0:1, :100]
    normalized = scaler.transform(sample_data)
    
    with torch.no_grad():
        reconstructed = autoencoder(torch.tensor(normalized, dtype=torch.float32)).numpy()
    
    denorm_original = scaler.inverse_transform(normalized)
    denorm_reconstructed = scaler.inverse_transform(reconstructed)
    
    plt.figure(figsize=(12,6))
    plt.plot(denorm_original.flatten(), label="Original")
    plt.plot(denorm_reconstructed.flatten(), linestyle="--", label="Reconstructed")
    plt.title(f"AutoEncoder Reconstruction (Iteration {iteration})")
    plt.legend()
    os.makedirs("./ae_reconstructions", exist_ok=True)
    plt.savefig(f"./ae_reconstructions/recon_{iteration}.png")
    plt.close()

def ae_power_generation(autoencoder, scaler, latent_samples):
    with torch.no_grad():
        generated = autoencoder.decoder(torch.tensor(latent_samples, dtype=torch.float32)).numpy()
    
    generated = scaler.inverse_transform(generated)
    generated[generated < 0] = 0
    return generated

def save_npy(old_data,num_samples,output_size,program_name):
    inp_data=[]
    for i in range(num_samples**output_size):
        inp_data.append(pm.data1D_process_inp(program_name,i))
    inp_data=np.array(inp_data)
    inp_data=inp_data[:,100:200]
    final_data= np.concatenate((old_data,inp_data), axis=0)
    print("save end")
    return final_data,inp_data

def save_npy_ori(old_data,num_samples,output_size,program_name):
    inp_data=[]
    fit_data=[]
    for i in range(num_samples**output_size):
        inp_data.append(pm.data1D_process_inp(program_name,i))
        fit_data.append(pm.data1D_process_fit(program_name, i))
    inp_data=np.array(inp_data)
    fit_data=np.array(fit_data)
    new_data = np.concatenate((
        inp_data[:, 100:200],  # 选择输入数据中的特定列
        fit_data[:, :]   # 选择拟合数据中的特定列
    ), axis=1)
    final_data= np.concatenate((old_data,new_data), axis=0)
    print("save end")
    return final_data



def plot_latent_distribution(data, iteration, latent_size):
    plt.figure(figsize=(20, 5))
    for i in range(latent_size):
        plt.subplot(1, latent_size, i+1)
        plt.hist(data[:, i], bins=30, alpha=0.7, color='green', edgecolor='black')
        mean = np.mean(data[:, i])
        std = np.std(data[:, i])
        plt.title(f"Latent {i}: μ={mean:.2f}, σ={std:.2f}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"./ae_data_distribution/latent_dist_{iteration}.png")
    plt.close()

def process_task(index, new_dir,Laser_grid):
    pm.generate_input_data1D(new_dir, index,Laser_grid[index])
    pm.run_command_1D(new_dir, index)

def thread_task(index, new_dir, Laser_grid):
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(process_task, index, new_dir, Laser_grid)

# Main workflow
input_size = 100
latent_size = 5  # Match the feature space dimension
num_samples = 5
num_iterations = 5

# Initialize AutoEncoder
autoencoder = AutoEncoder(input_size, latent_size)

# Load initial data
data_ori = np.load("original_data1.npy")
data=data_ori[:,:100]
program_name = 'Multi-1D-AE'
new_dir = pm.init1D(program_name)

power_scaler = MinMaxScaler()
power_scaler.fit_transform(data)

for iteration in range(num_iterations):
    print(f"\n=== Iteration {iteration+1} ===")
    
    # 1. Train AutoEncoder
    autoencoder = train_autoencoder(autoencoder, data, iteration,power_scaler)
    
    # 2. Reconstruct and plot
    reconstruct_and_plot(autoencoder, power_scaler, iteration)
    
    # 3. Generate new samples
    latent_mean, latent_std = calculate_latent_distribution(data, autoencoder, power_scaler)
    latent_samples = generate_latent_samples(latent_mean, latent_std, num_samples)
    Laser_grid = ae_power_generation(autoencoder, power_scaler, latent_samples)
    
    # 4. Create input format
    new_column = np.full((Laser_grid.shape[0], 100), 0.05747)
    new_column[:, 0] = 0
    Laser_grid = np.hstack((new_column, Laser_grid))
    
    # 5. Run simulations
    with ProcessPoolExecutor(max_workers=50) as pool:
        futures = [pool.submit(thread_task, i, new_dir, Laser_grid) for i in range(Laser_grid.shape[0])]
    
    # 6. Save new data
    data, new_data = save_npy(data, num_samples,latent_size,program_name)  # Assuming same index vector
    data_ori= save_npy_ori(data_ori, num_samples,latent_size,program_name)

    with torch.no_grad():
        reconstructed = autoencoder(torch.tensor(new_data, dtype=torch.float32)).numpy()
    plot_latent_distribution(reconstructed, iteration,latent_size)

# Final save
np.save("ae_combined_data", data_ori)
print("AutoEncoder workflow completed!")