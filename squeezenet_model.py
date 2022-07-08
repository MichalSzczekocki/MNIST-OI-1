import torch
import torchvision
import matplotlib.pyplot as plt
import time
from torch import nn, optim, utils
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from sklearn.metrics import confusion_matrix

import models.squeezenet
from functions import plot_confusion_matrix, test_and_show_errors, calculate_accuracy, validate, get_all_preds
from models import squeezenet

# Print
test = 0

loss_list = []

iteration_list = []

accuracy_list = []

count = 0

stop = False

stop_value = 0.0005

PATH = 'squeezenet_model_1.pt'

# Parametaers
batch_size = 30

learning_rate = 0.001

num_epochs = 1

momentum = 0.5

transform = transforms.Compose([
	transforms.Resize(224),
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x.expand(3, -1, -1))
])

# Data load
dataset_train = datasets.MNIST(
	'data', train=True, download=True, transform=transform)

dataset_test = datasets.MNIST(
	'data', train=False, download=True, transform=transform)

train_loader = utils.data.DataLoader(
	dataset_train, batch_size=batch_size, shuffle=False)

test_loader = utils.data.DataLoader(
	dataset_test, batch_size=batch_size, shuffle=False)

model = models.squeezenet.create_squeezenet_model()
#model = torch.hub.load('pytorch/vision:v0.9.0',
					   #'squeezenet1_0', pretrained=False)

# Load model
# model.load_state_dict(torch.load(PATH))

# Use graphic card if available
cuda = torch.cuda.is_available()

print("Is cuda available ?", cuda)

dev = "cuda" if cuda else "cpu"

# Choose device cuda or cpu
device = torch.device(dev)

# Create model
model = model.to(device)

# Draw model summary
summary(model, input_size=(3, 224, 224), device=dev)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Setup loss function
error = nn.CrossEntropyLoss()

# Setup scheduler (provides several methods to adjust the learning rate based on the number of epochs)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Freeze the features[5] layer
# print(model.features[5])

# model.features[5].expand1x1.weight.requires_grad = False
# model.features[5].expand1x1.bias.requires_grad = False

# model.features[3].expand1x1.weight.requires_grad = False
# model.features[3].expand1x1.bias.requires_grad = False

start_time = time.time()

for epoch in range(num_epochs):
	# Tells model that is going to be trained
	model.train()

	if stop:
		break

	# In this case data = images and target = labels
	for batch_idx, (data, target) in enumerate(train_loader):

		# Transfer to GPU or CPU allows to generate your data on multiple cores in real time
		data, target = data.to(device), target.to(device)

		# In PyTorch, we need to set the gradients to zero before starting to do backpropragation
		optimizer.zero_grad()

		# Feed network
		output = model(data)

		# Calculate loss
		loss = error(output, target)

		# Calculate change for each od weights and biases in model
		loss.backward()

		# Update weight and biases for example, the SGD optimizer performs: x += -lr * x.grad
		optimizer.step()

		count += 1

		if loss.data.cpu() <= stop_value:
			print("Stop condition achieved loss.data", stop_value)
			stop = True

		if (batch_idx + 1) % 10 == 0:
			# Switch to eval mode
			model.eval()

			with torch.no_grad():
				end_time = time.time()
				#accuracy = 0
				accuracy = float(validate.validate(
					model, device, train_loader))

				print("It took {:.2f} seconds to execute this".format(
					end_time - start_time))

				# Store loss, iteration and accuracy
				loss_list.append(loss.data.cpu())
				iteration_list.append(count)
				accuracy_list.append(accuracy)

				print("Epoch:", epoch + 1, "Batch:", batch_idx + 1, "Loss:",
					  float(loss.data), "Accuracy:", accuracy, "%")

			 # Switch to tain mode
			model.train()

		if stop:
			break

	# Adjust learning rate
	scheduler.step()

# SAVE MODEL
torch.save(model.state_dict(), PATH)

# VISUALIZATION LOSS AND ACCURACY

plt.plot(iteration_list, loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

plt.plot(iteration_list, accuracy_list, color="red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()


# CALCULATE MODEL ACCURACY ON TRAIN AND TEST DATA

calculate_accuracy.calculate_model_accuracy(
	model, device, test_loader, "Final test data")
calculate_accuracy.calculate_model_accuracy(
	model, device, train_loader, "Final train data")


# DRAW CONFUSION MATRIX

train_preds = get_all_preds.get_all_preds(model, train_loader, device)
cm = confusion_matrix(dataset_train.targets, train_preds.argmax(dim=1))
plot_confusion_matrix.plot_confusion_matrix(cm, dataset_train.classes,
											title="Confusion matrix train data")
plt.show()

test_preds = get_all_preds.get_all_preds(model, test_loader, device)
cm = confusion_matrix(dataset_test.targets, test_preds.argmax(dim=1))
plot_confusion_matrix.plot_confusion_matrix(cm, dataset_test.classes,
											title="Confusion matrix test data")
plt.show()


# SHOW WRONGLY CLASSIFIED PICTURES

test_and_show_errors.test_and_show_errors(model, device, dataset_test)