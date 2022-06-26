import torch 
import numpy as np
from tqdm import tqdm
    
class AMP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AMP, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        pass 
        # if isinstance(module, torch.nn.Linear):
        #     module.weight.data.normal_(mean=0.0, std=0.1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))

        return outputs

    def private_loss():
        pass

def train_amp(args, data):
    """
    args: command line arguments 
    data: tuple consisting of (train_x, train_y) for training 
    """
    # Get training data and convert to PyTorch tensor 
    # train_x, train_y = data
    # train_x = torch.Tensor(train_x)
    # train_y = torch.tensor(np.argmax(train_y, axis=1), dtype=torch.float)

    # model = AMP(input_dim=train_x.size(1),  output_dim=1)

    # criterion = torch.nn.BCELoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    pass 

    for _ in tqdm(range(args.epochs), desc="Training AMP Progress", leave=False, colour="yellow"):
        continue
        # optimizer.zero_grad()

        # outputs = torch.squeeze(model(train_x))

        # loss = criterion(outputs, train_y)
        # loss.backward()

        # optimizer.step()

    if args.debug:
        pass 
        # outputs = torch.squeeze(model(train_x))
        # outputs[outputs >= 0.5] = 1 
        # outputs[outputs < 0.5] = 0 
        
        # error_rate = torch.sum(torch.abs(torch.sub(outputs, train_y))).item() / outputs.size(0)
        
        # print("Training Accuracy: {:.2f}%".format((1 - error_rate) * 100))

    return # model

def find_loss_lr(x, y, model, poisoned=False):
    pass 
    # criterion = torch.nn.BCELoss()
    # x, y = torch.squeeze(torch.Tensor(x)), torch.Tensor(np.argmax(y, axis=1))
    
    # loss = criterion(model(x), y)

    # print("{} Loss:{:.3f}".format("Poisoned" if poisoned else "Unpoisoned", loss.item()))
