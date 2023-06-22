from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import SGD

torch.manual_seed(1)
def train(model, device, train_loader, optimizer, epoch,train_acc,train_losses,runName,L1flag=False):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.nll_loss(y_pred, target)


        # L1 Regularization
        if L1flag:
            l1_lambda = 1.0e-5
            l1_loss = torch.tensor(0., requires_grad=True)
            l1_loss=l1_loss.to(device)
            for name, param in model.named_parameters():
                l1_loss = l1_loss + l1_lambda*(torch.norm(param, 1))
            loss=loss+l1_loss

        train_losses[runName].append(loss.item())
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc[runName].append(100*correct/processed)
    return train_acc,train_losses


def test(model, device, test_loader, test_acc,test_losses,runName):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses[runName].append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc[runName].append(100. * correct / len(test_loader.dataset))
    return test_acc,test_losses,test_loss