import argparse

import time

import numpy
import pandas

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32, 16)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 2)
        self.activation3 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        return x

def get_data_loader(train_kwargs, test_kwargs):
    df = pandas.read_csv('../data/train.csv').values
    slice_index = int(df.shape[0] * 0.8)
    df_train = df[:slice_index, :]
    df_test = df[slice_index:, :]
    df_train_x = df_train[:, :32]
    df_train_y = df_train[:, 32:34]
    df_test_x = df_test[:, :32]
    df_test_y = df_test[:, 32:34]

    dl_train = DataLoader(
        TensorDataset(
            torch.tensor(df_train_x).float(),
            torch.tensor(df_train_y).float()),
        **train_kwargs)
    
    dl_test = DataLoader(
        TensorDataset(
            torch.tensor(df_test_x).float(),
            torch.tensor(df_test_y).float()),
        **test_kwargs)
    
    return dl_train, dl_test

def train(data_loader, model, optimizer, loss_function, device, epoch, log_interval, dry_train):
    model.train()
    for batch_index, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

        if batch_index % log_interval == 0 and not dry_train:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(x), len(data_loader.dataset),
                100. * batch_index / len(data_loader), loss.item()
            ))

def test(data_loader, model, loss_function, epoch):
    model.eval()
    loss = 0
    corret = 0
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(data_loader):
            # print(f"!!! + {batch_index}, {x.size()}, {y.size()}")
            y_hat = model(x)
            loss += loss_function(y_hat, y)
            pred = (y_hat > 0.55).float()
            # print(f"!!!!! + {pred}")
            corret += pred.eq(y.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)

    print(f'Test: Average loss: {loss}, Accuracy: {corret}/{2*len(data_loader.dataset)} ({100.0*corret/(2*len(data_loader.dataset))}%)')

def main():
    # arguments
    
    parser = argparse.ArgumentParser(description="s3e18 V0")
    parser.add_argument('--batch_size', type=int, default=64, metavar='B')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='B')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--no-model-saving', action='store_true', default=False)
    parser.add_argument('--dry-train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # data preprocessing
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = get_data_loader(train_kwargs, test_kwargs)

    # epoch

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.BCELoss()

    local_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)

    if not args.test:
        for epoch in range(1, args.epochs + 1):
            train(train_loader, model, optimizer, loss_function, device, epoch, args.log_interval, args.dry_train)
            test(test_loader, model, loss_function, epoch)

        if not args.no_model_saving:
            torch.save(model.state_dict(), "./state_dicts/s3e18_" + formatted_time)
            torch.save(model.state_dict(), "./state_dicts/s3e18_latest")
    else:
        model.load_state_dict(torch.load("./state_dicts/s3e18_latest"))
        df = pandas.read_csv("../data/test.csv").values
        y_hat = model(torch.tensor(df).float())
        pred = (y_hat > 0.55).float()

        save_file = torch.cat((torch.tensor(df)[:, 0].unsqueeze(1), pred), dim=1)
        df = pandas.DataFrame(save_file.numpy())
        df.columns = ["id", "EC1", "EC2"]
        df = df.astype(int)
        df.to_csv('./predictions/pred_' + formatted_time + ".csv", index=False)

    
if __name__ == "__main__":
    main()