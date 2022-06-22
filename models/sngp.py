import math
import copy
from statistics import mean
import pdb
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from torch import Tensor
import models.resnetsn

def RandomFeatureLinear(i_dim, o_dim, bias=True, require_grad=False):
    m = nn.Linear(i_dim, o_dim, bias)
    # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/bert_sngp.py
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    # freeze weight
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2. * math.pi)
        # freeze bias
        m.bias.requires_grad = require_grad
    return m


class SNGP(nn.Module):
    def __init__(self, backbone,
                 hidden_size=2048,
                 gp_kernel_scale=1.0,
                 num_inducing=1024,
                 gp_output_bias=0.,
                 layer_norm_eps=1e-12,
                 scale_random_features=True,
                 normalize_input=True,
                 gp_input_dim=128,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1e-3,
                 epochs=40,
                 num_classes=3,
                 device='cpu'):
        super(SNGP, self).__init__()
        self.backbone = backbone
        self.final_epochs = epochs - 1
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum

        self.pooled_output_dim = hidden_size
        
        self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2. / float(num_inducing))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input

        self._gp_input_layer = nn.utils.spectral_norm(nn.Linear(hidden_size, gp_input_dim, bias=False))
        # freeze weight
        self._gp_input_layer.weight.requires_grad = False

        self._gp_input_normalize_layer = torch.nn.LayerNorm(gp_input_dim, eps=layer_norm_eps)
        self._gp_output_layer = nn.Linear(num_inducing, num_classes, bias=False)
        # bert gp_output_bias_trainable is false
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L69
        self._gp_output_bias = torch.tensor([self.gp_output_bias] * num_classes).to(device)
        self._random_feature = RandomFeatureLinear(gp_input_dim, num_inducing)

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        self.initial_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing).to(device))
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)


    def gp_layer(self, gp_inputs, update_cov=True):
        # Supports lengthscale for custom random feature layer by directly
        # rescaling the input.
        # pdb.set_trace()
        gp_inputs = self._gp_input_layer(gp_inputs)
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        gp_feature = self._random_feature(gp_inputs)
        # cosine
        gp_feature = torch.cos(gp_feature)

        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if update_cov:
            # update precision matrix
            self.update_cov(gp_feature)
        return gp_feature, gp_output

    def reset_cov(self):
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def update_cov(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L346
        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = torch.matmul(gp_feature.t(), gp_feature)
        # Updates the population-wise precision matrix.
        if self.gp_cov_momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.gp_cov_momentum * self.precision_matrix +
                    (1. - self.gp_cov_momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        #self.precision_matrix.weight = precision_matrix_new
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)

    def compute_predictive_covariance(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix

    def forward(self, inputs, return_gp_cov: bool = False,
                update_cov: bool = True):
        # pdb.set_trace()
        x = self.backbone(inputs)
        
        # pdb.set_trace()
        gp_feature, gp_output = self.gp_layer(x, update_cov=update_cov)
        if return_gp_cov:
            gp_cov_matrix = self.compute_predictive_covariance(gp_feature)
            return gp_output, gp_cov_matrix
        # pdb.set_trace()
        return gp_output
