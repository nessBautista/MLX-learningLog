import gzip
import os
import pickle
from urllib import request
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TORCH_MNIST_MLP:
    def __init__(
        self,
        num_layers=2,
        hidden_dim=32,
        num_classes=10,
        batch_size=256,
        num_epochs=10,
        learning_rate=1e-1,
        seed=0,
        device=None
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = {
            'train_loss': [],
            'test_accuracy': [],
            'epoch_times': []
        }
        
    def mnist(
        self,
        save_dir="/tmp",
        base_url="https://raw.githubusercontent.com/fgnt/mnist/master/",
        filename="mnist.pkl",
    ):
        def download_and_save(save_file):
            filename = [
                ["training_images", "train-images-idx3-ubyte.gz"],
                ["test_images", "t10k-images-idx3-ubyte.gz"],
                ["training_labels", "train-labels-idx1-ubyte.gz"],
                ["test_labels", "t10k-labels-idx1-ubyte.gz"],
            ]

            mnist = {}
            for name in filename:
                out_file = os.path.join("/tmp", name[1])
                request.urlretrieve(base_url + name[1], out_file)
            for name in filename[:2]:
                out_file = os.path.join("/tmp", name[1])
                with gzip.open(out_file, "rb") as f:
                    mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
            for name in filename[-2:]:
                out_file = os.path.join("/tmp", name[1])
                with gzip.open(out_file, "rb") as f:
                    mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
            with open(save_file, "wb") as f:
                pickle.dump(mnist, f)

        save_file = os.path.join(save_dir, filename)
        if not os.path.exists(save_file):
            download_and_save(save_file)
        with open(save_file, "rb") as f:
            mnist = pickle.load(f)

        def preproc(x):
            return x.astype(np.float32) / 255.0

        mnist["training_images"] = preproc(mnist["training_images"])
        mnist["test_images"] = preproc(mnist["test_images"])
        
        # Convert to PyTorch tensors
        return (
            torch.FloatTensor(mnist["training_images"]),
            torch.LongTensor(mnist["training_labels"].astype(np.uint32)),
            torch.FloatTensor(mnist["test_images"]),
            torch.LongTensor(mnist["test_labels"].astype(np.uint32)),
        )

    class MLP(nn.Module):
        def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
            layers = []
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(nn.Linear(idim, odim))
                if idim != layer_sizes[-2]:  # Don't add ReLU after last layer
                    layers.append(nn.ReLU())
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    def train(self):
        # Load the data
        train_images, train_labels, test_images, test_labels = self.mnist()
        
        # Create data loaders
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Initialize model if not already initialized
        if self.model is None:
            self.model = self.MLP(
                self.num_layers, 
                train_images.shape[1], 
                self.hidden_dim, 
                self.num_classes
            ).to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for e in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            tic = time.perf_counter()
            
            # Training
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            # Evaluation
            accuracy = self.evaluate(test_loader)
            toc = time.perf_counter()
            epoch_time = toc - tic
            
            # Store metrics
            self.metrics['train_loss'].append(np.mean(epoch_losses))
            self.metrics['test_accuracy'].append(accuracy)
            self.metrics['epoch_times'].append(epoch_time)
            
            print(
                f"Epoch {e}: Test accuracy {accuracy:.3f}, "
                f"Train loss {self.metrics['train_loss'][-1]:.3f}, "
                f"Time {epoch_time:.3f} (s)"
            )
            
        return self.model
    
    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            return self.model(X)

    def get_metrics(self):
        """Return the training metrics."""
        return self.metrics

if __name__ == "__main__":
    # Example usage
    mlp = MNIST_MLP()
    mlp.train() 