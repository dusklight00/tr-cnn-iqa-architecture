import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
  def __init__(self, dim):
    super(CNN, self).__init__()

    self.dim_x = dim[0]
    self.dim_y = dim[1]

    # Load the pre-trained ResNet-18 model
    resnet = models.resnet18(pretrained=True)

    # Modify the first layer to accept grayscale images
    self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])

    # Replace the remaining layers
    self.bn1 = resnet.bn1
    self.relu = resnet.relu
    self.maxpool = resnet.maxpool
    self.layer1 = resnet.layer1
    self.layer2 = resnet.layer2
    self.layer3 = resnet.layer3
    self.layer4 = resnet.layer4
    self.avgpool = resnet.avgpool

    # Replace the last fully connected layer
    num_features = resnet.fc.in_features
    self.fc = nn.Linear(num_features, self.dim_x * self.dim_y)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    output = self.fc(x)
    output = output.view(-1, 1, self.dim_x, self.dim_y)
    return output