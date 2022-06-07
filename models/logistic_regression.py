import torch 
import numpy as np
from tqdm import tqdm
    
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
        # self.apply(self._init_weights)

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
    # train_y = torch.tensor(train_y, dtype=torch.long)

    model = LogisticRegression(input_dim=train_x.size(1),  output_dim=1)

    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for _ in tqdm(range(args.epochs), desc="Training LogisticRegression Progress", leave=False, colour="blue"):
        optimizer.zero_grad()

        outputs = torch.squeeze(model(train_x))

        loss = criterion(outputs, train_y)
        loss.backward()

        optimizer.step()

    if args.debug:
        # outputs = torch.argmax(model(train_x), dim=1)
        outputs = torch.squeeze(model(train_x))
        outputs[outputs >= 0.5] = 1 
        outputs[outputs < 0.5] = 0 
        
        error_rate = torch.sum(torch.abs(torch.sub(outputs, train_y))).item() / outputs.size(0)
        
        print("Training Accuracy: {:.2f}%".format((1 - error_rate) * 100))

    return model

def find_loss_lr(x, y, model, poisoned=False):
    criterion = torch.nn.BCELoss()
    x, y = torch.squeeze(torch.Tensor(x)), torch.Tensor(np.argmax(y, axis=1))
    
    loss = criterion(model(x), y)

    print("{} Loss:{:.3f}".format("Poisoned" if poisoned else "Unpoisoned", loss.item()))
