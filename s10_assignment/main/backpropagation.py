from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import SGD

torch.manual_seed(1)


train_losses = []
test_losses = []
train_acc = []
test_accuracies = []
train_loss = []
train_accuracies = []

def get_correct_count(prediction, labels):
    return prediction.argmax(dim=1).eq(labels).sum().item()


def get_incorrect_preds(prediction, labels):
    prediction = prediction.argmax(dim=1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()

def train(model, device, lr_scheduler, criterion, train_loader, optimizer, epoch):

    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        # if l1 > 0:
        #     loss += l1 * sum(p.abs().sum() for p in model.parameters())

        train_loss += loss.item() * len(data)

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += get_correct_count(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Batch_id={batch_idx}')
        lr_scheduler.step()

    train_acc = 100 * correct / processed
    train_loss /= processed
    train_accuracies.append(train_acc)
    train_losses.append(train_loss)
    print(f"\nTrain Average Loss: {train_losses[-1]:0.2f}%")
    print(f"Train Accuracy: {train_accuracies[-1]:0.4f}")



def test(model, device, criterion, test_loader):

    model.eval()

    test_loss = 0
    test_loss1 = 0
    correct = 0
    processed = 0
    test_acc=0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)

            test_loss += criterion(pred, target).item() * len(data)

            correct += get_correct_count(pred, target)
            processed += len(data)


    test_acc = 100 * correct / processed
    test_loss /= processed
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)
    print(f"Test Average loss: {test_loss:0.4f}")
    print(f"Test Accuracy: {test_acc:0.2f}")