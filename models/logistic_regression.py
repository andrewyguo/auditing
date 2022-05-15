import torch 
import numpy as np
from tqdm import tqdm
    
class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)

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
    train_x= torch.Tensor(train_x)
    train_y = torch.tensor(np.argmax(train_y, axis=1), dtype=torch.long)

    model = LogisticRegression(input_dim=train_x.size(1),  output_dim=2)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(args.epochs), desc="Training LogisticRegression Progress", leave=False):
        optimizer.zero_grad()

        outputs = model(train_x)

        loss = criterion(outputs, train_y)

        loss.backward()

        optimizer.step()

    if args.debug:
        outputs = torch.argmax(model(train_x), dim=1)
        error_rate = torch.sum(torch.abs(torch.sub(outputs, train_y))).item() / outputs.size(0)
        
        print("Model Accuracy on Training Set: {}".format((1 - error_rate) * 100))

    return model
