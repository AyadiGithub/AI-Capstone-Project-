"""
Crack detection in buildings
"""

"""
The dataset contains concrete images having cracks. The data is collected from various METU Campus Buildings.
The dataset is divided into two as negative and positive crack images for image classification.
Each class has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels.
The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016).
High-resolution images have variance in terms of surface finish and illumination conditions.
No data augmentation in terms of random rotation or flipping is applied.

Citations:
2018 – Özgenel, Ç.F., Gönenç Sorguç, A.
“Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings”,
ISARC 2018, Berlin.

Lei Zhang , Fan Yang , Yimin Daniel Zhang, and Y. J. Z., Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016).
Road Crack Detection Using Deep Convolutional Neural Network.
In 2016 IEEE International Conference on Image Processing (ICIP). http://doi.org/10.1109/ICIP.2016.7533052
"""

##############################################################################
"""
Dataset Challenges:
    Cracks can be confused as background texture noise or foreign objects.
    Inhomogeneous Illumination
    Other irregularities
"""
"""
Dataset Link:
    https://data.mendeley.com/datasets/5y9wdsg2zt/2
"""

##############################################################################

# In[1] Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas
import os
import glob
import numpy as np
from torch import optim
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[2] Config

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# In[3] Loading Negative Images

# Directory for image files
directory = r'C:'

# Negative images (No cracks)
negative = 'Negative'

# Negative file path
negative_file_path = os.path.join(directory, negative)
negative_file_path

# Use function os.listdir to print a list of files in the directory
os.listdir(negative_file_path)[0:10]  # first 10 files in the list

# Create full image path for files in the negative folder
[os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path)][0:10]

#  make sure the files are .jpg extension using the method 'endswith()'
# Create a list of all negative image files (str) in the directory
negative_files = [os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()

# In[4] Loading Positive Images

positive = "Positive"

# Positive file path
positive_file_path = os.path.join(directory, positive)

# Show positive files
os.listdir(positive_file_path)[0:10]

# Create a list of all positive image files in the directory
positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()

# In[5] View images

# Using PIL
positive_image1 = Image.open(positive_files[0])
positive_image2 = Image.open(positive_files[1])

negative_image1 = Image.open(negative_files[0])
negative_image2 = Image.open(negative_files[1])

plt.imshow(negative_image1)
plt.title("1st Image With No Cracks")
plt.show()

plt.imshow(positive_image1)
plt.title("1st Image With Cracks")
plt.show()

# count number of files in each directory
negative_count = len(negative_files)
negative_count

positive_count = len(positive_files)
positive_count

# In[6] Data Preparation

number_of_samples = len(negative_files) + len(positive_files)
print("Total number of images: ", number_of_samples)

# Assign lables to images, 1 for positve, 0 for negative
Y = torch.zeros([number_of_samples])  # Create zeros tensor
Y = Y.type(torch.LongTensor)
print("Type of Tensor Y: ", Y.dtype)
# labeling starts with positive images with 2 steps and negative images with 2 steps
# Labels assigned in alternating manner
Y[0::2] = 1
Y[1::2] = 0
Y

# Filling list with all positive and negative images
all_files = []
all_files = [None] * number_of_samples
all_files[0::2] = positive_files
all_files[1::2] = negative_files

# Print first 4 samples
for y, file in zip(Y, all_files[0:4]):
    plt.imshow(Image.open(file))
    plt.title("y = " + str(y.item()))
    plt.show()

"""
Training and Validation Split
"""
train = True
if train:
    all_files = all_files[0:30000]
    Y = Y[0:30000]
else:
    all_files = all_files[30000:]
    Y = Y[30000:]

# In[7] Create Dataset Class and object


class Dataset(Dataset):

    # Constructor
    def __init__(self, transform=None, train=True):

        directory = r'C:'
        positive = "Positive"
        negative = "Negative"

        positive_file_path = os.path.join(directory, positive)
        negative_file_path = os.path.join(directory, negative)
        positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files = [os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()
        # Index
        self.all_files = [None]*number_of_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files
        # The transform is goint to be used on image
        self.transform = transform
        # Torch.LongTensor
        self.Y = torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2] = 1
        self.Y[1::2] = 0

        if train:
            self.all_files = self.all_files[0:30000]
            self.Y = self.Y[0:30000]
            self.len = len(self.all_files)
        else:
            self.all_files = self.all_files[30000:]
            self.Y = self.Y[30000:]
            self.len = len(self.all_files)

    # Get the length
    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):

        image = Image.open(self.all_files[idx])
        y = self.Y[idx]

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y

# In[8] Create Dataset


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create transform ToTensor, Normalize transform and compose them
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Create Dataset train and validation object
dataset_train = Dataset(transform=transform, train=True)
dataset_val = Dataset(transform=transform, train=False)

# Shape of Image
dataset_train[0][0].shape

# Image size
size_of_image = 3*227*227
output_dim = 2

# In[9] Create Model, Loss Function, Optimizer, Parameters

learning_rate = 0.1
momentum = 0.3
batch_size = 1000

# Softmax Class from nn.Module


class SoftMax(nn.Module):

    # Constructor
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    # Prediction
    def forward(self, x):
        x = self.linear(x)
        return x


model = SoftMax(size_of_image, output_dim)
model.to(device)
print("The Model:\n", model)

# Lets see the initialized parameters and their size
print('W: ', list(model.parameters())[0].size())
print('b: ', list(model.parameters())[1].size())
print("The Parameters are: \n", model.state_dict())

# Optimizer. SGD with momentum
optimizer = optim.SGD(model.parameters(), momentum=momentum, lr=learning_rate)

# Criterion loss function
criterion = nn.CrossEntropyLoss()

# Dataloader Train and Val
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size)
val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size)

# In[10] Train the model

# Train the model
epochs = 10
LOSS_List = []  # Empty list to store LOSS
Accuracy_List = []
N_test = len(dataset_val)

# Training function


def train_model(n_epochs):
    for epoch in range(epochs):

        for x, y in train_loader:

            x, y = x.to(device), y.to(device)

            model.train()

            optimizer.zero_grad()

            z = model(x.view(-1, size_of_image))  # Reshaping to 28x28

            loss = criterion(z, y)

            loss.backward()

            optimizer.step()

        correct = 0

        print(correct)

        # Perform a prediction on the validation data
        for x_test, y_test in val_loader:

            x_test, y_test = x_test.to(device), y_test.to(device)

            model.eval()

            z = model(x_test.view(-1, size_of_image))

            _, yhat = torch.max(z.data, 1)  # Take max value from z

            correct += (yhat == y_test).sum().item()

            print(correct)

        accuracy = correct / N_test

        LOSS_List.append(loss.data)

        Accuracy_List.append(accuracy)


train_model(epochs)


# In[11] Plot Accuracy and Loss

"""
Analyze the model
"""

# Plot the loss and accuracy
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(LOSS_List, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('total loss', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.plot(Accuracy_List, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()
