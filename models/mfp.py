# Multi View Feature Propagation
# created by Etzion Harari | TAU

# imports
from tqdm import tqdm
from .fp import FP
import torch

# Globals
TNSR = type(torch.tensor([]))

# simple functions
to_torch = lambda mat, device='cpu': (mat if type(mat) == TNSR else torch.from_numpy(mat)).to(torch.device(device))
cat_union = lambda tensors_list: torch.cat(tensors_list, 1)
mean_union = lambda tensors_list: torch.mean(torch.stack(tensors_list).float(), dim=0)


# MFP object
class MFP:

    # Implementation of MFP
    # Multi View Feature Propagation Algorithm for attributed graphs

    def __init__(self,
                 iters=100,
                 views=10,
                 use_tqdm=False,
                 union_method=cat_union,
                 features_ratio=.8,
                 add_noise=True,
                 mean=0,
                 std=1):
        """
        Init object

        :iters: number of propagation iterations (gamma)
        :views: number of views (eta)
        :features_ratio: ratio of sampling retain features (p)
        :use_tqdm: use tqdm to the propagation calculation
        :add_noise: add gaussian noise to instead empty features (instead of zeros)
        :mean: mean for gaussian noise (mu)
        :std: std for gaussian noise (sigma)
        """

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.process = tqdm if use_tqdm else lambda x: x
        self.features_ratio = features_ratio
        self.union_function = union_method
        self.add_noise = add_noise
        self.iters = iters
        self.views = views

        self.mean = mean
        self.std = std

    def stochastic_sampling(self, features, missing_indices):
        """
        Sampeling Function
        Remove some of the retain features and keep only subset of them

        :features: feature matrix
        :missing_indices: indices of missing features
        """

        # calculate feature indices to sampled
        sampled_indices = torch.rand(features.shape).to(self.device) < self.features_ratio
        exist_features = (((missing_indices == False) * 1.) * (sampled_indices * 1.)) > 0
        missing_values = torch.normal(self.mean, self.std, features.shape).to(self.device) if self.add_noise else 0

        # sample subset
        sampled_features = torch.where(exist_features, features, missing_values).to(torch.float32).to(self.device)
        indices = exist_features == False

        # return sampled_features and missing features indices
        return sampled_features, indices

    def multi_view_propagation(self, features, missing_indices, edges):
        """
        Multi view Feature Propagation function
        create self.views views, each with subset of the retain features.
        each view propagate using the FP algorithm, than aggregate to multi view representation

        :features: input features
        :missing_indices: indices of missing features
        :edges: edges of the graph
        """

        propagated_matrices = []

        # for each iteration defined by the user:
        for _ in self.process(range(self.views)):
            # sample self.features_ratio from features
            sampled_matrix, indices = self.stochastic_sampling(features, missing_indices)

            # feature propagation
            propagated_features = FP(index=indices).prop(sampled_matrix, edges)

            # add the propagated results to the matrices list
            propagated_matrices.append(to_torch(propagated_features, device=self.device))

        # returned concat results
        return self.union_function(propagated_matrices)

    def prop(self, features, indices, edges):
        """
        appling MFP on given dataset

        :features: input features
        :indices: indices of missing features
        :edges: edges of the graph
        """

        # convert inputs to torch tensors
        features = to_torch(features, device=self.device)
        indices = to_torch(indices, device=self.device)
        edges = to_torch(edges, device=self.device)

        # apply MFP
        united_propagated_features = self.multi_view_propagation(features, indices, edges)

        # return output
        return united_propagated_features

# created by Etzion Harari | TAU