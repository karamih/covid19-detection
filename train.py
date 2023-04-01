import torch
import torch.nn as nn
from torch import optim

from data_setup import creat_dataloader
from model_builder import model
from engin import train
from utils import visualization, save_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dir = 'covid19-dataset/train'
test_dir = 'covid19-dataset/test'
BATCH_SIZE = 8
EPOCHS = 15
learning_rate = 0.001

train_dataloader, test_dataloader, classes = creat_dataloader(train_dir=train_dir, test_dir=test_dir, batch_size=BATCH_SIZE)


co19_model = model(len(classes)).to(device)

optimizer = optim.Adam(co19_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


results = train(model=co19_model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=EPOCHS,
                device=device)


visualization(results=results)

save_model(model=co19_model, dir_name='saved_model', file_name='covid19_detector.pth')