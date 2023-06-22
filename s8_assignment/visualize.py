import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision

def plot_misclassified_images(model, test_loader, classes, device):

    def unnormalize(img):
      channel_means = (0.4914, 0.4822, 0.4471)
      channel_stdevs = (0.2469, 0.2433, 0.2615)
      img = img.numpy().astype(dtype=np.float32)

      for i in range(img.shape[0]):
          img[i] = (img[i]*channel_stdevs[i])+channel_means[i]

      return np.transpose(img, (1,2,0))



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


def losses_plot(train_losses,test_losses):
    plt.style.use('seaborn-poster')

    fig, ax = plt.subplots(1, 2)

    color1 = 'blue'
    color2 = 'red'
    color3 = 'green'
    linestyle1 = '-'
    linestyle2 = '--'
    linestyle3 = '-.'
    EPOCHS=20

    train_epoch_linspace = np.linspace(1, EPOCHS, len(train_losses['GN']))
    test_epoch_linspace = np.linspace(1, EPOCHS, len(test_losses['GN']))
    #Left plot

    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Training Loss', color=color1)
    ax[0].plot(train_epoch_linspace, train_losses['GN'], color=color1, linestyle=linestyle1,alpha=0.5, label='Group Norm')
    ax[0].plot(train_epoch_linspace, train_losses['LN'], color=color2, linestyle=linestyle2,alpha=0.7, label='Layer Norm')
    ax[0].plot(train_epoch_linspace, train_losses['BN'], color=color3, linestyle=linestyle3,alpha=0.9, label='Batch Norm')
    ax[0].tick_params(axis='y', labelcolor=color1)
    ax[0].legend(loc='upper right')
    #Right plot

    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Test Loss', color=color2)
    ax[1].plot(test_epoch_linspace, test_losses['GN'], color=color1, linestyle=linestyle1,alpha=0.5, label='Group Norm')
    ax[1].plot(test_epoch_linspace, test_losses['LN'], color=color2, linestyle=linestyle2,alpha=0.7, label='Layer Norm')
    ax[1].plot(test_epoch_linspace, test_losses['BN'], color=color3, linestyle=linestyle3,alpha=0.9, label='Batch Norm')
    ax[1].tick_params(axis='y', labelcolor=color2)
    ax[1].legend(loc='upper right')

    fig.tight_layout()
    fig.set_size_inches(16, 5)
    plt.show()


def accuracy_plot(train_acc,test_acc):
    plt.style.use('seaborn-poster')

    fig, ax = plt.subplots(1, 2)

    color1 = 'blue'
    color2 = 'red'
    color3 = 'green'
    linestyle1 = '-'
    linestyle2 = '--'
    linestyle3 = '-.'
    EPOCHS=20
    train_epoch_linspace = np.linspace(1, EPOCHS, len(train_acc['GN']))
    test_epoch_linspace = np.linspace(1, EPOCHS, len(test_acc['GN']))
    #Left plot

    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Training Accuracy', color=color1)
    ax[0].plot(train_epoch_linspace, train_acc['GN'], color=color1, linestyle=linestyle1,alpha=0.5, label='Group Norm')
    ax[0].plot(train_epoch_linspace, train_acc['LN'], color=color2, linestyle=linestyle2,alpha=0.7, label='Layer Norm')
    ax[0].plot(train_epoch_linspace, train_acc['BN'], color=color3, linestyle=linestyle3,alpha=0.9, label='Batch Norm')
    ax[0].tick_params(axis='y', labelcolor=color1)
    ax[0].legend(loc='lower right')
    #Right plot

    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Test Accuracy', color=color2)
    ax[1].plot(test_epoch_linspace, test_acc['GN'], color=color1, linestyle=linestyle1,alpha=0.5, label='Group Norm')
    ax[1].plot(test_epoch_linspace, test_acc['LN'], color=color2, linestyle=linestyle2,alpha=0.7, label='Layer Norm')
    ax[1].plot(test_epoch_linspace, test_acc['BN'], color=color3, linestyle=linestyle3,alpha=0.9, label='Batch Norm')
    ax[1].tick_params(axis='y', labelcolor=color2)
    ax[1].legend(loc='lower right')

    fig.tight_layout()
    fig.set_size_inches(16, 5)
    plt.show()

def show_images(trainloader, classes):
    # Function to show an image
    def imshow(img, figsize=(10, 5)):
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.axis('off')  # Remove axis
        plt.show()

    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images in a single line
    grid_image = torchvision.utils.make_grid(images, nrow=len(images))

    # Show the grid image with a larger size
    imshow(grid_image, figsize=(12, 8))  # Adjust the figsize to make the image bigger

    # Print labels
    print('      '.join('%6s' % classes[labels[j]] for j in range(len(images))))