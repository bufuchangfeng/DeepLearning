import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


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
    def __init__(self, x, y, phase):
        self.x = x
        self.phase = phase

        self.y = y if phase != 'test' else None

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        if self.phase != 'test':
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class MNIST_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes):
        super(MNIST_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        pass


def main():
    model = MNIST_LSTM(in_dim=784, hidden_dim=256, num_layers=2, num_classes=10)

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = MNIST_DATASET(x_train, y_train, 'train')
    valid_dataset = MNIST_DATASET(x_valid, y_valid, 'valid')
    test_dataset = MNIST_DATASET(x_test, y_test, 'test')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for index, images, labels in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test

if __name__ == '__main__':
    main()