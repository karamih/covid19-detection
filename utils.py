import torch
from path import Path
import matplotlib.pyplot as plt


def acc_fn(pred, true):
    correct = torch.eq(pred, true).sum().item()
    acc = (correct / len(true)) * 100
    return acc


def save_model(model: torch.nn.Module,
               dir_name: str,
               file_name: str):
    dir_path = Path(dir_name)
    dir_path.mkdir(parents=True, exist_ok=True)

    save_model_path = dir_path / file_name

    torch.save(model.state_dict(), save_model_path)


def visualization(results):
    plt.subplot(121)
    plt.plot(results['train loss'], c='red', label='train loss')
    plt.plot(results['validation loss'], c='blue', label='validation loss')
    plt.title('LOSS')
    plt.legend()

    plt.subplot(122)
    plt.plot(results['train acc'], c='red', label='train acc')
    plt.plot(results['validation acc'], c='blue', label='validation acc')
    plt.title('ACCURACY')
    plt.legend()

    plt.show()



