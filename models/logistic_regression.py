import torch 
import numpy as np
from tqdm import tqdm

from .model_utils import calculate_accuracy

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))

        return outputs

def train_lr(args, data):
    """
    args: command line arguments 
    data: tuple consisting of (train_x, train_y) for training 
    """
    # Get training data and convert to PyTorch tensor 
    train_x, train_y = data
    train_x = torch.Tensor(train_x)
    train_y = torch.tensor(np.argmax(train_y, axis=1), dtype=torch.float)

    model = LogisticRegression(input_dim=train_x.size(1),  output_dim=1)

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(args.epochs), desc="Training LogisticRegression Progress", leave=False, colour="blue"):
        optimizer.zero_grad()

        outputs = torch.squeeze(model(train_x))

        loss = criterion(outputs, train_y)
        loss.backward()

        optimizer.step()
        if epoch % 5 == 0 and args.debug:
            calculate_accuracy(epoch, model, train_x, train_y)

    if args.debug:
        calculate_accuracy(args.epochs, model, train_x, train_y)

    return model

def find_loss_lr(x, y, model, poisoned=False):
    criterion = torch.nn.BCELoss()
    x, y = torch.squeeze(torch.Tensor(x)), torch.Tensor(np.argmax(y, axis=1))
    
    loss = criterion(model(x), y)

    print("{} Loss:{:.3f}".format("Poisoned" if poisoned else "Unpoisoned", loss.item()))
