import csv

import help as hp
import model as md
import torch
from torch import nn
from torch import optim

# Define Processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Parameters
img_dim = 28
hidden_sizes = [100, 800, 600]
output_size = 10

# Fetch Datasets
train_tranformer = hp.train_transform(img_dim)
test_tranformer = hp.test_transform(img_dim)

trainloader = hp.download_data('data/MNIST_data/', train_tranformer, True, 128)
testloader = hp.download_data('data/MNIST_data/', test_tranformer, False, 128)

# Fetch Model
model = md.model(img_dim * img_dim, hidden_sizes, output_size)
# model = md.Net(output_size)
model.to(device)

# Train Parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
epochs = 30
model_path = 'data/best_model.pt'
saved_model_device = torch.device("cpu")

# Train Model
train_accs, test_accs, losses = hp.train_loop(model, epochs, trainloader, testloader, optimizer, criterion,
                                              model_path, saved_model_device, device, img_dim)
# save results
acc_loss_path = 'data/acc_loss.csv'
records = {'train_acc': train_accs, 'test_acc': test_accs, 'loss': losses}
with open(acc_loss_path, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(records.keys())
    writer.writerows(zip(*records.values()))

# Plot losses and accuracies
hp.plotLoss(losses)
hp.plotAccuracy(train_accs, test_accs)
