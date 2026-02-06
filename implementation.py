!pip install torch torchvision matplotlib seaborn numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time

# ==========================================
# 1. Custom Implementation of Adam Optimizer
# ==========================================
class AdamFromScratch(torch.optim.Optimizer):
    """
    Implements Adam algorithm from the paper 'Adam: A Method for Stochastic Optimization'.
    
    Parameters:
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamFromScratch, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Gradient at time t (g_t)
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')

                # Apply weight decay (L2 regularization) if specified
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (m_0 = 0)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values (v_0 = 0)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                t = state['step']

                # Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first moment estimate: m_hat_t = m_t / (1 - beta1^t)
                bias_correction1 = 1 - beta1 ** t
                m_hat = exp_avg / bias_correction1

                # Compute bias-corrected second raw moment estimate: v_hat_t = v_t / (1 - beta2^t)
                bias_correction2 = 1 - beta2 ** t
                v_hat = exp_avg_sq / bias_correction2

                # Update parameters: theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
                denom = v_hat.sqrt().add_(group['eps'])
                
                step_size = group['lr']
                
                # In-place update
                p.data.addcdiv_(m_hat, denom, value=-step_size)

        return loss

# ==========================================
# 2. Experimental Setup (Data & Model)
# ==========================================

def get_data():
    """Download MNIST and prepare loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Using MNIST as a proxy for larger image datasets for demonstration speed
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                             download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                            download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron for MNIST classification."""
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(optimizer_class, learning_rate, train_loader, epochs=3, name="Adam"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    
    if name == "AdamFromScratch":
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    else:
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    model.train()
    print(f"Starting training with {name}...")
    
    total_steps = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                losses.append(loss.item())
                total_steps += 1
        
        print(f"Epoch {epoch+1}/{epochs} completed.")
        
    return losses

# ==========================================
# 3. Execution and Visualization
# ==========================================
if __name__ == "__main__":
    train_loader, test_loader = get_data()
    
    # Compare our implementation of Adam vs Standard SGD
    print("Training Custom Adam...")
    adam_losses = train_model(AdamFromScratch, lr=0.001, train_loader=train_loader, epochs=2, name="AdamFromScratch")
    
    print("Training Standard SGD...")
    sgd_losses = train_model(torch.optim.SGD, lr=0.01, train_loader=train_loader, epochs=2, name="SGD")
    
    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Downsample for cleaner plot if needed, or plot all
    plt.plot(adam_losses, label='Adam (Custom Impl.)', alpha=0.8)
    plt.plot(sgd_losses, label='SGD (PyTorch Std.)', alpha=0.8)
    
    plt.title('Convergence Comparison: Adam vs SGD on MNIST')
    plt.xlabel('Training Steps (x10)')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.show()