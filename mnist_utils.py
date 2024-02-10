# mnist_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(batch_size=64):
    """
    Load the MNIST dataset and create data loaders for training and testing.

    Parameters:
    - batch_size (int): The batch size for data loaders. Default is 64.

    Returns:
    - trainloader (torch.utils.data.DataLoader): Data loader for the training set.
    - testloader (torch.utils.data.DataLoader): Data loader for the test set.

    This function loads the MNIST dataset using torchvision.datasets.MNIST. It applies
    transforms to normalize the pixel values to the range [-1, 1]. It then creates data
    loaders for the training and test sets using torch.utils.data.DataLoader. The training
    data loader shuffles the data, while the test data loader does not shuffle the data.

    Example Usage:
    trainloader, testloader = load_mnist(batch_size=128)
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def create_simple_nn():
    """
    Create a simple neural network model for image classification.

    Returns:
    - model (torch.nn.Module): The neural network model.

    This function defines a simple neural network model for image classification tasks.
    The model consists of a sequence of layers:
    - Flatten layer: Reshapes the input image tensor into a 1D tensor.
    - Fully connected (Linear) layer: Converts the flattened input into a hidden representation.
      The input size is 28*28 (MNIST image size) and the output size is 128.
    - ReLU activation function: Applies element-wise ReLU activation to introduce non-linearity.
    - Fully connected (Linear) layer: Converts the hidden representation into class probabilities.
      The input size is 128 (output of the previous layer) and the output size is 10 (number of classes).

    Example Usage:
    model = create_simple_nn()
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model

def train_model(model, trainloader, optimizer, criterion, epochs=10):
    """
    Train a neural network model using the provided data loader, optimizer, and loss function.

    Parameters:
    - model (torch.nn.Module): The neural network model to train.
    - trainloader (torch.utils.data.DataLoader): Data loader for the training dataset.
    - optimizer (torch.optim.Optimizer): Optimizer to update the model parameters.
    - criterion (torch.nn.Module): Loss function to compute the training loss.
    - epochs (int): Number of epochs for training. Default is 10.

    Returns:
    - history (dict): Dictionary containing training history (loss and accuracy).

    This function trains a neural network model using the provided data loader, optimizer, and
    loss function. It iterates over the specified number of epochs and updates the model parameters
    based on the training data. At each epoch, it computes the average loss and accuracy, and stores
    them in a dictionary called 'history'. The 'history' dictionary contains two lists: 'loss' and
    'accuracy', which track the training loss and accuracy over epochs, respectively.

    Example Usage:
    history = train_model(model, trainloader, optimizer, criterion, epochs=10)
    """
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs.view(-1, 28*28))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')
    return history

def evaluate_model(model, testloader):
    """
    Evaluate a trained neural network model on the provided test dataset.

    Parameters:
    - model (torch.nn.Module): The trained neural network model to evaluate.
    - testloader (torch.utils.data.DataLoader): Data loader for the test dataset.

    Returns:
    - accuracy (float): Accuracy of the model on the test dataset.

    This function evaluates a trained neural network model on the provided test dataset.
    It computes the accuracy of the model by comparing its predictions with the ground-truth labels.
    The accuracy is calculated as the ratio of correct predictions to the total number of samples.
    If an error occurs during evaluation (e.g., division by zero), it prints an error message and
    returns None.

    Example Usage:
    accuracy = evaluate_model(model, testloader)
    """
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs.view(-1, 28*28))
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')
        return accuracy
    except ZeroDivisionError:
        print("Error: Division by zero. No test samples or empty testloader.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def get_model_and_optimizer(optimizer_name, lr=0.001):
    """
    Create a neural network model and corresponding optimizer based on the specified parameters.

    Parameters:
    - optimizer_name (str): Name of the optimizer to use ('adam', 'adagrad', or 'rmsprop').
    - lr (float): Learning rate for the optimizer (default: 0.001).

    Returns:
    - model (torch.nn.Module): The created neural network model.
    - optimizer (torch.optim.Optimizer): The optimizer for training the model.

    This function creates a neural network model using the create_simple_nn function.
    It then selects the appropriate optimizer based on the provided optimizer_name and learning rate.
    The supported optimizers are Adam, Adagrad, and RMSprop.
    If an invalid optimizer name is provided, it raises a ValueError.
    
    Example Usage:
    model, optimizer = get_model_and_optimizer('adam', lr=0.001)
    """
    model = create_simple_nn()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name. Choose 'adam', 'adagrad', or 'rmsprop'.")
    return model, optimizer

