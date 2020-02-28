import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .classification_base import ClassificationModel

class NonConvClassificationModel(ClassificationModel):
    def __init__(self, input_size: int, output_size: int):
        super(NonConvClassificationModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        return F.log_softmax(y, dim=1)

    def _transform_input(self, input_data: np.ndarray):
        return input_data.view(input_data.shape[0], -1)

    @property
    def num_classes(self):
        return self.fc3.out_features


class ConvClassificationModel(ClassificationModel):
    def __init__(self):
        super(ConvClassificationModel, self).__init__()
        self.filter_size = 5
        self.conv1 = nn.Conv2d(1, 3, self.filter_size)
        self.conv2 = nn.Conv2d(3, 5, self.filter_size)

        self.fc1 = nn.Linear(4*4*self.filter_size, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Convolutional Filter Learning
        y = self.conv1(x)
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(y)
        y = self.conv2(y)
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(y)
        
        # Classification
        y = y.view(-1, 4*4*self.filter_size)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        return F.log_softmax(y, dim=1)

    def _transform_input(self, input_data: np.ndarray):
        return input_data

    @property
    def num_classes(self):
        return self.fc2.out_features