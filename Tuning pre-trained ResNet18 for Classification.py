"""

Tuning pre-trained ResNet18 for Classification

"""

# In[1] Imports
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os


# In[2] Config

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# In[3] Create Dataset Class and object

"""
Dataset Links:
    https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip
    https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip
"""


class Dataset(Dataset):

    # Constructor
    def __init__(self, transform=None, train=True):
        directory = r'C:/'
        positive = "Positive_tensors"
        negative = "Negative_tensors"

        positive_file_path = os.path.join(directory, positive)
        negative_file_path = os.path.join(directory, negative)
        positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files = [os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        number_of_samples = len(positive_files) + len(negative_files)
        self.all_files = [None] * number_of_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files

        self.transform = transform

        self.Y = torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2] = 1  # 1 for positive classification
        self.Y[1::2] = 0  # 0 for negative classificaiton

        if train:
            self.all_files = self.all_files[0:30000]
            self.Y = self.Y[0:30000]
            self.len = len(self.all_files)
        else:
            self.all_files = self.all_files[30000:]
            self.Y = self.Y[30000:]
            self.len = len(self.all_files)

    # Length
    def __len__(self):
        return self.len

    # Get item
    def __getitem__(self, idx):

        image = torch.load(self.all_files[idx])
        y = self.Y[idx]

        # Apply transforms on image, if any
        if self.transform:
            image = self.transform(image)

        return image, y


print("Dataset class creation Done!")

# In[4] Load pre-trained model ResNet18 and edit output layer

model = models.resnet18(pretrained=True)
model.to(device)

train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)

print("Dataset object creation Done!")

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 2).to(device)

print(model)

# In[6] Create Datalodder objects, Loss function, Optimizer

# Train loader and validation_loader
train_loader = DataLoader(dataset=train_dataset, batch_size=64)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=32)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr=0.001)

# In[7] Train Model

# Train the model
n_epochs = 1
loss_list = []
accuracy_list = []
correct = 0
N_test = len(validation_dataset)
N_train = len(train_dataset)
start_time = time.time()
Loss = 0

# Training function


def train_model(n_epochs):
    for epoch in range(n_epochs):

        for x, y in train_loader:

            x, y = x.to(device), y.to(device)

            model.train()

            optimizer.zero_grad()  # Clear gradient

            z = model(x)  # Make a prediction

            loss = criterion(z, y)  # calculate loss

            loss.backward()  # Backward propagation

            optimizer.step()  # Update parameters

            loss_list.append(loss.data)

        correct = 0

        print(correct)

        # Perform a prediction on the validation data
        for x_test, y_test in validation_loader:

            x_test, y_test = x_test.to(device), y_test.to(device)

            model.eval()

            z = model(x_test)

            _, yhat = torch.max(z.data, 1)  # Take max value from z

            correct += (yhat == y_test).sum().item()

            print(correct)

        accuracy = correct / N_test

        loss_list.append(loss.data)

        accuracy_list.append(accuracy)

        # Accuracy
        print("Accuracy = ", accuracy)


train_model(n_epochs)


# In[8] Evaluation

# Plot iteration vs loss
plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

# Plot function


def show_data(data_sample):
    plt.imshow(data_sample[0].cpu().numpy().reshape(224, 224, 3), cmap='coolwarm')
    plt.title('y = ' + str(data_sample[1].item()))


# Plot misclassified examples

Softmax_function = nn.Softmax(dim=-1)
count = 0
N_samples = 0
for x, y in validation_dataset:
    x, y = x.to(device), y.to(device)  # For GPU Training
    z = model(x.reshape(224, 3, 32, 7))
    yhat = torch.max(z, 1)
    if yhat != y:
        show_data((x, y))
        plt.show()
        print("probability of class ", torch.max(Softmax_function(z)).item())
        count += 1
    if count >= 4:
        break


# Plot the first 5 misclassified samples and their respective probability by using torch.max
for x_test, y_test in validation_loader:

    x_test, y_test = x_test.to(device), y_test.to(device)  # For GPU Training

    model.eval()

    z = model(x_test)

    _, yhat = torch.max(z.data, 1)  # Take max value from z
    for i in range(len(y_test)):
        count += 1
        if yhat[i] != y_test[i]:
            print("Sample#: %d - Predicted Value: %d - Actual Value: %d" % (count, yhat[i], y_test[i]))
            N_samples += 1
            if N_samples >= 4:
                break
    if N_samples >= 4:
        break

print("done!")
