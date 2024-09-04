import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50*4*4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50*4*4)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class SimpleMILAttention(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleMILAttention, self).__init__()

        # Feature extractor (a simple CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Flatten the output of the feature extractor
        self.flatten = nn.Flatten()

        # Define the attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # For binary classification (adjust if using softmax for multi-class)
        )

    def forward(self, x):
        # x: [batch_size, num_patches, channels, height, width]
        batch_size, num_patches, channels, height, width = x.size()
        
        # Process each patch through the feature extractor
        x = x.view(batch_size * num_patches, channels, height, width)
        H = self.feature_extractor(x)
        H = self.flatten(H)  # H: [batch_size * num_patches, feature_dim]

        # Reshape to separate patches
        H = H.view(batch_size, num_patches, -1)  # H: [batch_size, num_patches, feature_dim]

        # Apply attention mechanism
        A = self.attention(H)  # A: [batch_size, num_patches, 1]
        A = torch.transpose(A, 2, 1)  # A: [batch_size, 1, num_patches]
        A = F.softmax(A, dim=2)  # Softmax over the patches

        # Weighted sum of instance features
        M = torch.bmm(A, H)  # M: [batch_size, 1, feature_dim]
        M = M.squeeze(1)  # M: [batch_size, feature_dim]

        # Classification
        Y_prob = self.classifier(M)  # Y_prob: [batch_size, num_classes]
        Y_hat = torch.ge(Y_prob, 0.5).float()  # Binarize output for binary classification

        return Y_prob, Y_hat

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        return neg_log_likelihood.mean()
