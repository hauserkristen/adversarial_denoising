import abc
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer

class ClassificationModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(ClassificationModel, self).__init__()

    @abc.abstractmethod
    def _transform_input(self, input_data: np.ndarray):
        raise NotImplementedError('Define input transformation.')

    def train(self, train_data: torch.utils.data.DataLoader, loss_func: nn.Module, optimizer: Optimizer):
        super(ClassificationModel, self).train()
        
        for data, label in train_data:
            # Zero grad
            optimizer.zero_grad()
            
            # Predict
            transformed_data = self._transform_input(data)
            pred_probabilities = self(transformed_data)
            
            # Calculate loss (assumes probatility and index format)
            loss = loss_func(pred_probabilities, label)
            loss.backward()
            
            # Take a step
            optimizer.step()

    def eval(self, test_data: torch.utils.data.DataLoader):
        super(ClassificationModel, self).eval()
        
        with torch.no_grad():
            num_correct = 0.0
            
            for data, label in test_data:
                # Predict
                transformed_data = self._transform_input(data)
                pred_probabilities = self(transformed_data)
                
                # Identify max prediction probability
                pred_label = pred_probabilities.data.max(1, keepdim=True)[1]
                
                # Count the number correct
                num_correct += pred_label.eq(label.data.view_as(pred_label)).sum()

            return (float(num_correct) / len(test_data.dataset))


    def load(self, weight_dict):
        model_state_dict = {key: torch.tensor(value, dtype=torch.float) for key, value in weight_dict.items()}
        self.load_state_dict(model_state_dict, strict=False)

    def save(self):
        model_state_dict = self.model.state_dict()
        model_numpy = {key: value.numpy() for key, value in model_state_dict.items()}
        return model_numpy
