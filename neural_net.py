# neural_net.py - architektúra ResNet pre Duck Chess v štýle AlphaZero.

import torch
import torch.nn as nn
import torch.nn.functional as F


# Jeden reziduálny blok
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# Vstup:  (batch, 16, 8, 8)
# Výstup: policy logits (batch, 4416)  +  hodnota (batch, 1)
class DuckChessNet(nn.Module):
    def __init__(self, num_res_blocks=6, channels=128):
        super().__init__()

        self.conv_input = nn.Sequential(
            nn.Conv2d(16, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList(
            [ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Hlava politiky (policy head)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4416)
        )

        # Hlava hodnoty (value head)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv_input(x)
        for block in self.res_blocks:
            out = block(out)
        return self.policy_head(out), self.value_head(out)

    @torch.no_grad()
    def predict(self, observation, legal_actions, device):
        # Pre MCTS: jeden observation → (policy, value)
        self.eval()
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        policy_logits, value = self.forward(obs)
        policy_logits = policy_logits.squeeze(0)

        # Maskujeme nelegálne ťahy
        mask = torch.full((4416,), float('-inf'), device=device)
        for a in legal_actions:
            mask[a] = 0.0
        policy = F.softmax(policy_logits + mask, dim=0)

        return policy.cpu().numpy(), value.item()