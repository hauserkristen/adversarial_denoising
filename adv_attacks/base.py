import abc
import torch
import matplotlib.pyplot as plt

class AdversarialAttack(object):
    @abc.abstractmethod
    def attack(self, data: torch.Tensor, data_gradient):
        raise NotImplementedError('Must implement attack.')

    def run(self, net: torch.nn.Module, test_data: torch.utils.data.DataLoader, loss_func, visualize: bool = False):
        # Accuracy counter
        correct = 0
        adv_examples = []

        # Loop over all examples in test set
        for data, label in test_data:
            # Data grad is required for most attacks (including FGSM)
            data.requires_grad = True

            # Predict
            pred_probabilities = net.classify(data)
            pred_label = pred_probabilities.data.max(1, keepdim=True)[1]

            # Check for initial accuracy
            if pred_label.item() != label.item():
                continue

            # Calculate loss (assumes probatility and index format)
            net.zero_grad()
            loss = loss_func(pred_probabilities, label)
            loss.backward()

            # Collect data gradient
            data_grad = data.grad.data

            # Call attack
            perturbed_data = self.attack(net, data, label, data_grad)

            # Plot
            if visualize:
                f, ax = plt.subplots(1,3)
                img_data = data.detach().numpy()[0,0,:,:]
                per_data = perturbed_data.detach().numpy()[0,0,:,:]
                per = per_data - img_data
                ax[0].imshow(img_data, cmap='gray')
                ax[1].imshow(per_data, cmap='gray')
                ax[2].imshow(per, cmap='gray')
                plt.show()

            # Predict
            adv_pred_probabilities = net.classify(perturbed_data)
            adv_pred_label = adv_pred_probabilities.data.max(1, keepdim=True)[1]

            # Check for success
            if adv_pred_label.item() == label.item():
                correct += 1

        # Calculate final accuracy for this epsilon
        accuracy = correct / float(len(test_data))

        return accuracy