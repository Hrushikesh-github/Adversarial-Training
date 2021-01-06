import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet50
import json
import numpy as np
import torch.optim as optim

# Read the image, resize to 224 and convert to PyTorch Tensor
pig_img = Image.open("pig.jpg")
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    ])
# Have the tensor in PyTorch standards -> batch_size * num_channels * height * width
pig_tensor = preprocess(pig_img)[None, :, :, :]

# A class to normalize the image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

# Values are standard normalizaion for ImageNet images
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# load pre-trained ResNet50 and put into evaluation mode
model = resnet50(pretrained=True)
model.eval()

# form predictions, outputs a 1000d vector
pred = model(norm(pig_tensor))

with open("Downloads/imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
print(imagenet_classes[pred.max(dim=1)[1].item()])

# 341 class corresponding to 'hog'
cross_entropy_loss_hog = nn.CrossEntropyLoss()(model(norm(pig_tensor)),torch.LongTensor([341])).item()
print(cross_entropy_loss_hog)
print("Probablity: ", np.exp(cross_entropy_loss_hog))

# The epsilon value used to set the max,min values of delta, our perturbation
epsilon = 2. / 255
delta = torch.zeros_like(pig_tensor, requires_grad=True)

# Initialize SGD optimizer, adding the delta is important here
opt = optim.SGD([delta], lr=1e-1)

# Loop 30 times, performing gradient descent wrt delta
for i  in range(30):
    pred = model(norm(pig_tensor + delta))
    loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([341]))
    if i % 5 == 0:
        print(i, loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()
    # Clamp delta value with epsilon, so that we are within the range
    delta.data.clamp_(-epsilon, epsilon)

print("True class probability: ", nn.Softmax(dim=1)(pred)[0, 341].item())

max_class = pred.max(dim=1)[1].item()
print("Predicted class: ", imagenet_classes[max_class])
print("Predicted probability: ", nn.Softmax(dim=1)(pred)[0, max_class].item())

plt.imshow((pig_tensor + delta)[0].detach().numpy().transpose(1,2,0))
plt.show()

# Zoom in delta by a factor of 50 and show image
plt.imshow((50*delta+0.5)[0].detach().numpy().transpose(1,2,0))
plt.show()
plt.imshow((delta+0.5)[0].detach().numpy().transpose(1,2,0))

plt.show()

# Targeted Learning
delta = torch.zeros_like(pig_tensor, requires_grad=True)
opt = optim.SGD([delta], lr=5e-3)

for t in range(100):
    pred = model(norm(pig_tensor + delta))
    loss = (-nn.CrossEntropyLoss()(pred, torch.LongTensor([341])) +
            nn.CrossEntropyLoss()(pred, torch.LongTensor([404])))
    if t % 10 == 0:
        print(t, loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()
    delta.data.clamp_(-epsilon, epsilon)

max_class = pred.max(dim=1)[1].item()
print("Predicted class: ", imagenet_classes[max_class])
print("Predicted probability:", nn.Softmax(dim=1)(pred)[0,max_class].item())

