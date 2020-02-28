import torch
import numpy as np

from .base import AdversarialAttack

class OnePixel(AdversarialAttack):
    def __init__(self, num_iters: int, population_size: int, num_pixels: int):
        self.iterations = num_iters
        self.pop_size = population_size
        self.num_pixels = num_pixels

    def attack(self, net: torch.nn.Module, data: torch.Tensor, label: torch.Tensor, data_gradient: torch.Tensor):
        min_log_p = np.log(1.0/net.num_classes)

        # Generate initial candidates
        data_dim = data.size()
        candidate_loc = np.array([self.generate_candidate(data_dim) for i in range(self.pop_size)])
        candidate_val = np.random.rand(self.pop_size, self.num_pixels)

        # Evaluate
        pred_probability = self.evaluate_candidates(net, data, label, candidate_loc, candidate_val)
        
        # Run iterations
        for iteration in range(self.iterations):
            # Early Stopping
            if pred_probability.min() < min_log_p:
                break

            # Generate new candidate
            new_candidate_loc = np.array([self.generate_candidate(data_dim) for i in range(self.pop_size)])
            new_candidate_val = np.random.rand(self.pop_size, self.num_pixels)

            # Evaluate new candidate
            new_pred_probability = self.evaluate_candidates(net, data, label, new_candidate_loc, new_candidate_val)

            # Take best of size population
            all_candidate_loc = np.vstack((candidate_loc, new_candidate_loc))
            all_candidate_val = np.vstack((candidate_val, new_candidate_val))
            all_pred_probability = np.concatenate((pred_probability, new_pred_probability))

            # Replace old candidates with new ones where they are better
            successors = np.argpartition(all_pred_probability, self.pop_size)[:self.pop_size]
            candidate_loc = all_candidate_loc[successors]
            candidate_val = all_candidate_val[successors]
            pred_probability = all_pred_probability[successors]

        # Retrieve perturb data
        best_index = pred_probability.argmin()
        best_attack = self.perturb_data(data, candidate_loc[best_index], candidate_val[best_index])

        return best_attack

    def generate_candidate(self, dimensions):
        candidate = []
        for i in range(self.num_pixels):
            pixel = []
            for dim in dimensions:
                pixel.append(np.random.choice(dim))
            candidate.append(pixel)

        return candidate

    def perturb_data(self, data: torch.Tensor, candidate_loc: np.ndarray, candidate_val: np.ndarray):
        preturbed_data = data.clone()
        for i, pixel in enumerate(candidate_loc):
            preturbed_data[tuple(pixel)] = candidate_val[i]
        return preturbed_data

    def evaluate_candidates(self, net: torch.nn.Module, data: torch.Tensor, label: torch.Tensor, candidate_loc: np.ndarray, candidate_val: np.ndarray):
        pred_probabilities = []
        for i in range(self.pop_size):
            # Perturb data
            perturbed_data = self.perturb_data(data, candidate_loc[i], candidate_val[i])

            # Predict
            adv_pred_prob = net.classify(perturbed_data)[0,label]
            pred_probabilities.append(adv_pred_prob.item())
            
        return np.array(pred_probabilities)
