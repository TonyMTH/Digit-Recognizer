from torchvision import datasets, transforms
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt


def download_data(path, transformer, datatype, batch_size):
    # Download and load the training data
    dataset = datasets.MNIST(path, download=True, train=datatype, transform=transformer)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=True)
    return dataloader


def train_transform(dim):
    # Define a transform to normalize the data
    return transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        #                        transforms.RandomRotation(30),
        #                        transforms.RandomHorizontalFlip(),
        #                        transforms.Resize(dim),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5))
                               ])


def test_transform(dim):
    # Define a transform to normalize the data
    return transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        #                        transforms.Resize(dim),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5))
                               ])


# def train_transform():
#     # Define a transform to normalize the data
#     return transforms.Compose([transforms.RandomRotation(30),
#                                transforms.RandomResizedCrop(28),
#                                transforms.RandomHorizontalFlip(),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.5, 0.5, 0.5],
#                                                     [0.5, 0.5, 0.5])])
#
#
# def test_transform():
#     # Define a transform to normalize the data
#     return transforms.Compose([transforms.RandomResizedCrop(28),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.5, 0.5, 0.5],
#                                                     [0.5, 0.5, 0.5])])


def train(loader, model, optimizer, criterion, device):
    running_loss = 0
    num_samples = 0

    for images, labels in loader:
        # Move Data to device
        images, labels = images.to(device), labels.to(device)

        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)

        optimizer.zero_grad()

        output = model.forward(images)  # 1) Forward pass
        loss = criterion(output, labels)  # 2) Compute loss
        loss.backward()  # 3) Backward pass
        optimizer.step()  # 4) Update model

        running_loss += loss.item()
        num_samples += output.size(0)

    return model, running_loss


def accuracy(loader, model, device, img_dim=28):
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            # Move Data to device
            images, labels = images.to(device), labels.to(device)

            # Flatten images long vector
            images.resize_(images.size()[0], img_dim * img_dim)

            scores = model(images)
            _, predictions = scores.max(1)

            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

    return float(num_correct) / float(num_samples) * 100


def train_loop(model, epochs, trainloader, testloader, optimizer, criterion, model_path,
               saved_model_device, device, img_dim):
    least_loss = np.inf

    train_accs = []
    test_accs = []
    losses = []

    for e in range(epochs):

        # Training
        model.train()
        model, loss = train(iter(trainloader), model, optimizer, criterion, device)

        # Accuracy
        model.eval()
        train_acc = accuracy(iter(trainloader), model, device, img_dim)
        test_acc = accuracy(iter(testloader), model, device, img_dim)

        # Save best model
        if loss < least_loss:
            least_loss = loss
            best_model_state = copy.deepcopy(model)
            best_model_state.to(saved_model_device)
            torch.save(best_model_state, model_path)

        # Collections
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        losses.append(loss)

        # Printing
        print(f"\tEpoch: {e + 1}/{epochs}\t Loss: {loss:.4f}\tTrain Acc: {train_acc:.4f}\tTest Acc: {test_acc:.4f}")

    return train_accs, test_accs, losses


def plotLoss(losses):
    plt.plot(losses)
    plt.title('Loss Function')
    plt.show()


def plotAccuracy(train_accs, test_accs):
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()
