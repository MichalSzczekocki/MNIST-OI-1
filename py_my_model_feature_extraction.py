import torch
import torchvision
import matplotlib.pyplot as plt
import time
from torch import nn, optim, utils
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from models import my_model_feature_extraction
from functions import plot_confusion_matrix, test_and_show_errors, get_all_preds, calculate_accuracy, validate
from functions import pytorchtool

test = 0
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
stop = False
#stop_value = 0.0005
stop_value = 0.05
PATH = 'custom_model_2.pt'

# hyperparameters
batch_size = 128
#batch_size = 32
learning_rate = 0.001
num_epochs = 10
momentum = 0.5

# data load
transform = transforms.Compose([transforms.ToTensor()])
dataset_train = datasets.MNIST(
    'data', train=True, download=True, transform=transform)
dataset_test = datasets.MNIST(
    'data', train=False, download=True, transform=transform)
train_loader = utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=False)
test_loader = utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False)

# use graphic card if available
cuda = torch.cuda.is_available()
print("Is cuda available ?", cuda)
dev = "cuda" if cuda else "cpu"

# choose device cuda or cpu
device = torch.device(dev)

# create model
model = my_model_feature_extraction.CNNModel().to(device)

# load model
# model.load_state_dict(torch.load(PATH))

# draw model summary
summary(model, input_size=(1, 28, 28), device=dev)

# optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# loss function setup
error = nn.CrossEntropyLoss()
# scheduler setup (provides several methods to adjust the learning rate based on the number of epochs)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# freeze the convolution_layer2 layer
# print(model)
# model.convolution_layer2.weight.requires_grad = False
# model.convolution_layer2.bias.requires_grad = False

start_time = time.time()
for epoch in range(num_epochs):
    # indicates model that is going to be trained
    model.train()
    if stop:
        break
    # in this case data = images and target = labels
    early_stopping = pytorchtool.EarlyStopping(patience=7, verbose=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        # transfer to GPU or CPU allows to generate your data on multiple cores in real time
        data, target = data.to(device), target.to(device)
        # in PyTorch, we need to set the gradients to zero before starting to do backpropragation
        optimizer.zero_grad()
        # feed network
        output = model(data)
        # calculate loss
        loss = error(output, target)
        # calculate change for each od weights and biases in model
        loss.backward()
        # update weight and biases for example, the SGD optimizer performs: x += -lr * x.grad
        optimizer.step()
        count += 1
        if loss.data.cpu() <= stop_value:
            print("Stop condition achieved loss.data", stop_value)
            stop = True
        if (batch_idx + 1) % 10 == 0:
            # switch to eval mode
            model.eval()
            with torch.no_grad():
                end_time = time.time()
                accuracy = float(validate.validate(
                     model, device, train_loader))
                print("It took {:.2f} seconds to execute this".format(
                    end_time - start_time))
                # store loss, iteration and accuracy
                loss_list.append(loss.data.cpu())
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                print("Epoch:", epoch + 1, "Batch:", batch_idx + 1, "Loss:",
                      float(loss.data), "Accuracy:", accuracy, "%")
            # switch to train mode
            model.train()
            early_stopping(float(loss.data), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if stop:
            break
    # adjust learning rate
    scheduler.step()

# save model
torch.save(model.state_dict(), PATH)

# loss and accuracy plots
plt.plot(iteration_list, loss_list, color="red")
plt.xlabel("number of iterations")
plt.ylabel("loss")
plt.title("Loss vs number of iterations")
plt.show()

plt.plot(iteration_list, accuracy_list, color="red")
plt.xlabel("number of iterations")
plt.ylabel("accuracy")
plt.title("Accuracy vs number of iterations")
plt.show()

# find model accuracy on train and test data
calculate_accuracy.calculate_accuracy(
    model, device, train_loader, "Final train data")
calculate_accuracy.calculate_accuracy(
    model, device, test_loader, "Final test data")


# plot confusion matrix
train_preds = get_all_preds.get_all_preds(model, train_loader, device)
cm = confusion_matrix(dataset_train.targets, train_preds.argmax(dim=1))
plot_confusion_matrix.plot_confusion_matrix(cm, dataset_train.classes,
                                            title="Confusion matrix for training data")
plt.show()

test_preds = get_all_preds.get_all_preds(model, test_loader, device)
cm = confusion_matrix(dataset_test.targets, test_preds.argmax(dim=1))
plot_confusion_matrix.plot_confusion_matrix(cm, dataset_test.classes,
                                            title="Confusion matrix for testing data")
plt.show()

# display wrongly classified images
test_and_show_errors.test_and_show_errors(model, device, dataset_test)
