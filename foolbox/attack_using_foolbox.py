import torch
import torchvision
import foolbox as fb
import PIL
import matplotlib.pyplot as plt

# Obtain the pretrained ResNet model, specify the preprocessing expected by the model and the bound of the input space
model = torchvision.models.resnet18(pretrained=True)
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)

model = model.eval()

# Turn your PyTorch model into Foolbox model
fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
# If model had different bounds, we do fmodel = fmodel.transform_bounds((0, 1))

# Provide a small set of sample images from different datasets
images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)

print(type(images))
print(images.shape)
fb.plot.images(images)
plt.show()

# Check the accuracy of model on our evaluation set
print(fb.utils.accuracy(fmodel, images, labels))

# Instantiate the attack class
attack = fb.attacks.LinfDeepFoolAttack()

(raw, clipped, is_adv) = attack(fmodel, images, labels, epsilons=0.03)
print(is_adv)
fb.plot.images(clipped - images, n=4, bounds=(0, 255), scale=50.)
plt.show()

