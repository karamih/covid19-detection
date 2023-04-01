# for creating train loop

import torch
from tqdm.auto import tqdm
from utils import acc_fn


def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim,
               loss_fn: torch.nn,
               device: torch.device):
    model.train()

    train_loss, train_acc = 0, 0

    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        train_acc += acc_fn(pred.argmax(dim=1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    return train_loss, train_acc


def valid_step(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn,
               device: torch.device):
    model.eval()

    valid_loss, valid_acc = 0, 0

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()
            valid_acc += acc_fn(pred.argmax(dim=1), y)

        valid_loss /= len(test_dataloader)
        valid_acc /= len(test_dataloader)

    return valid_loss, valid_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim,
          loss_fn: torch.nn,
          epochs: int,
          device: torch.device):
    result = {'train loss': [],
              'train acc': [],
              'validation loss': [],
              'validation acc': []}

    for epoch in tqdm(range(epochs)):
        print(f"epoch: {epoch + 1}/{epochs}")
        train_loss, train_acc = train_step(model=model,
                                           train_dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           device=device)

        valid_loss, valid_acc = valid_step(model=model,
                                           test_dataloader=test_dataloader,
                                           loss_fn=loss_fn,
                                           device=device)

        print(
            f"training loss: {train_loss:.5f} | training acc: {train_acc:.3f}% | validation loss: {valid_loss:.5f} | "
            f"validation acc: {valid_acc:.3f}%")

        result['train loss'].append(train_loss)
        result['train acc'].append(train_acc)
        result['validation loss'].append(valid_loss)
        result['validation acc'].append(valid_acc)

    return result