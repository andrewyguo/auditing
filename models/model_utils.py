import torch

def calculate_accuracy(epoch, model, x, y):
    outputs = torch.squeeze(model(x))
    outputs[outputs >= 0.5] = 1 
    outputs[outputs < 0.5] = 0 
    
    error_rate = torch.sum(torch.abs(torch.sub(outputs, y))).item() / outputs.size(0)
    
    print("Epoch {} Training Accuracy: {:.2f}%".format(epoch, (1 - error_rate) * 100))
