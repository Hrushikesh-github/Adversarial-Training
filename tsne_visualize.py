import numpy as np
from sklearn.manifold import TSNE
import torchvision
from torchvision import datasets, models, transforms
import torch
from imutils import paths
import sys
import matplotlib.pyplot as plt
#sys.path.append("/content/drive/My Drive/preprocessing")
#sys.path.append("/content/drive/My Drive/datasets")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

data_dir = '/home/hrushikesh/images/flower-17'

batch_size = 4

# A data loader where images are arranged in tree of directories and performs the required transformations
image_dataset = datasets.ImageFolder(data_dir, data_transforms)

# A dataloader to get data into batches
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=len(image_dataset), shuffle=False, num_workers=1)

print(len(image_dataset), torch.Tensor(image_dataset[0][0]).shape)

images, labels = next(iter(dataloader))
images = images.reshape(images.shape[0], -1).numpy()
#images = images.numpy()
print(images.shape)

X_embedded = TSNE(n_components=2).fit_transform(images)
print(X_embedded.shape)

c = []
for i in range(0, 240):
    if i < 80:
        c.append('r')
    elif i > 160:
        c.append('b')
    else:
        c.append('y')

for i in range(240):
    plt.scatter(X_embedded[i][0], X_embedded[i][1], alpha=0.8)
#plt.legend(label=['Bluebell', 'Sunflower', 'Tigerlily'])
plt.show()
