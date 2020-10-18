import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import copy


def load_data():
    data = loadmat("mnist_all.mat")

    # print(data.keys())

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for i in range(10):
        temp_df = pd.DataFrame(data["train" + str(i)])
        temp_df['label'] = i
        train_data = train_data.append(temp_df)
        temp_df = pd.DataFrame(data["test" + str(i)])
        temp_df['label'] = i
        test_data = test_data.append(temp_df)

    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_labels = np.array(train_data['label'])
    test_labels = np.array(test_data['label'])

    train_data = train_data.drop('label', axis=1)
    test_data = test_data.drop('label', axis=1)

    train_data = np.array(train_data) / 255
    test_data = np.array(test_data) / 255

    return train_data, test_data, train_labels, test_labels


BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train, x_test, y_train, y_test = load_data()

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.2, random_state=0)


class MNIST_DATASET(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        return self.transforms(self.x[index].reshape(28, 28)), self.y[index]

    def __len__(self):
        return len(self.x)


class LSTM_ATTENTION(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes):
        super(LSTM_ATTENTION, self).__init__()

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def attention(self, query, key, value):

        batch_size = value.shape[1]

        # value (28, BATCH_SIZE, 256)

        query = query[1, :, :].reshape(batch_size, 256, 1)  # query (BATCH_SIZE, 256, 1)

        key = key.permute(1, 0, 2)  # key (BATCH_SIZE, 28, 256)

        attention_weights = torch.bmm(key, query).reshape(batch_size, 28)  # attention_weights (BATCH_SZIE, 28)

        softmax_attention_weights = F.softmax(attention_weights, dim=1).reshape(batch_size, 1, 28)
        # softmax_attention_weights (BATCH_SIZE, 1, 28)

        value = value.permute(1, 0, 2)  # value (BATCH_SIZE, 28, 256)

        context = torch.bmm(softmax_attention_weights, value).reshape(batch_size, 256)

        return context

    def forward(self, x):
        # hidden = (torch.randn(2, 256, 256).double(),
        #           torch.randn(2, 256, 256).double())

        out, (h, c) = self.lstm(x)

        out = self.attention(h, out, out)

        x = self.classifier(out)
        return x


def main():
    model = LSTM_ATTENTION(in_dim=28, hidden_dim=256, num_layers=2, num_classes=10)
    model.to(DEVICE)
    model.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = MNIST_DATASET(x_train, y_train)
    valid_dataset = MNIST_DATASET(x_valid, y_valid)
    test_dataset = MNIST_DATASET(x_test, y_test)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    best_model = None
    best_acc = -1

    for epoch in range(EPOCHS):
        model.train()
        for index, (images, labels) in enumerate(train_loader):

            images = torch.squeeze(images)
            images = images.permute(1, 0, 2).double()

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if index % 100 == 0:
                print('Train Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, EPOCHS, loss.item()))

        model.eval()
        with torch.no_grad():
            valid_correct = 0
            for images, labels in valid_loader:
                images = torch.squeeze(images)
                images = images.permute(1, 0, 2).double()

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                valid_correct += (predicted == labels).sum().item()

            acc = 100.0 * valid_correct / len(valid_loader.dataset)

            print("Epoch: {} The accuracy of total {} images: {}%".format(epoch + 1, len(valid_loader.dataset),
                                                                          100.0 * valid_correct / len(
                                                                              valid_loader.dataset)))
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model)
                print('get new model!')

    model = copy.deepcopy(best_model)
    model.to(DEVICE)
    model.double()
    with torch.no_grad():
        test_correct = 0
        for images, labels in test_loader:
            images = torch.squeeze(images)
            images = images.permute(1, 0, 2).double()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            test_correct += (predicted == labels).sum().item()

    print("The accuracy of total {} images: {}%".format(len(valid_loader.dataset),
                                                        100.0 * valid_correct / len(valid_loader.dataset)))


# acc on test data 98.9%
if __name__ == '__main__':
    main()
