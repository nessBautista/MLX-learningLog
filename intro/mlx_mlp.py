import gzip
import os
import pickle
from urllib import request
import time
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class MLX_MNIST_MLP:
    def __init__(
        self,
        num_layers=2,
        hidden_dim=32,
        num_classes=10,
        batch_size=256,
        num_epochs=10,
        learning_rate=1e-1,
        seed=0,
        device=mx.cpu
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        
        # Set random seed and device
        np.random.seed(self.seed)
        mx.set_default_device(device)
        
        self.model = None
        self.optimizer = None
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
        return map(mx.array, (
            mnist["training_images"],
            mnist["training_labels"].astype(np.uint32),
            mnist["test_images"],
            mnist["test_labels"].astype(np.uint32),
        ))

    class MLP(nn.Module):
        def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
            self.layers = [
                nn.Linear(idim, odim)
                for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
            ]

        def __call__(self, x):
            for l in self.layers[:-1]:
                x = nn.relu(l(x))
            return self.layers[-1](x)

    def loss_fn(self, model, X, y):
        return nn.losses.cross_entropy(model(X), y, reduction="mean")

    def batch_iterate(self, X, y):
        perm = mx.array(np.random.permutation(y.size))
        for s in range(0, y.size, self.batch_size):
            ids = perm[s : s + self.batch_size]
            yield X[ids], y[ids]

    def evaluate(self, X, y):
        """Evaluate the model on the given data."""
        return mx.mean(mx.argmax(self.model(X), axis=1) == y)

    def train(self):
        # Load the data
        train_images, train_labels, test_images, test_labels = self.mnist()

        # Initialize model if not already initialized
        if self.model is None:
            self.model = self.MLP(self.num_layers, train_images.shape[-1], 
                                self.hidden_dim, self.num_classes)
            mx.eval(self.model.parameters())

        # Setup optimizer
        self.optimizer = optim.SGD(learning_rate=self.learning_rate)
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)

        @partial(mx.compile, inputs=self.model.state, outputs=self.model.state)
        def step(X, y):
            loss, grads = loss_and_grad_fn(self.model, X, y)
            self.optimizer.update(self.model, grads)
            return loss

        @partial(mx.compile, inputs=self.model.state)
        def eval_fn(X, y):
            return mx.mean(mx.argmax(self.model(X), axis=1) == y)

        # Training loop
        for e in range(self.num_epochs):
            epoch_losses = []
            tic = time.perf_counter()
            
            # Training
            for X, y in self.batch_iterate(train_images, train_labels):
                loss = step(X, y)
                mx.eval(self.model.state)
                epoch_losses.append(loss.item())
            
            # Evaluation
            accuracy = eval_fn(test_images, test_labels)
            toc = time.perf_counter()
            epoch_time = toc - tic
            
            # Store metrics
            self.metrics['train_loss'].append(np.mean(epoch_losses))
            self.metrics['test_accuracy'].append(accuracy.item())
            self.metrics['epoch_times'].append(epoch_time)
            
            print(
                f"Epoch {e}: Test accuracy {accuracy.item():.3f}, "
                f"Train loss {self.metrics['train_loss'][-1]:.3f}, "
                f"Time {epoch_time:.3f} (s)"
            )
            
        return self.model

    def get_metrics(self):
        """Return the training metrics."""
        return self.metrics

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        return self.model(mx.array(X))

if __name__ == "__main__":
    # Example usage
    mlp = MNIST_MLP()
    mlp.train()
