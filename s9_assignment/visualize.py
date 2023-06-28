import matplotlib.pyplot as plt
import torch
from dataset import unnormalize

# define a function to plot misclassified images
def plot_misclassified_images(model, test_loader, classes, device):
    # set model to evaluation mode
    model.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        npimg = unnormalize(misclassified_images[i].cpu())
        plt.imshow(npimg, cmap='gray', interpolation='none')
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()

def train_plot_stats(train_losses,train_accuracies):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(train_losses)
        axs[0].set_title("Training Loss")
        axs[1].plot(train_accuracies)
        axs[1].set_title("Training Accuracy")

def test_plot_stats(test_losses,test_accuracies):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(test_losses)
        axs[0].set_title("Test Loss")
        axs[1].plot(test_accuracies)
        axs[1].set_title("Testing Accuracy")