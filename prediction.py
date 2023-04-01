import torch
from model_builder import model
from PIL import Image
from torchvision import transforms


def predict(img):
    torch.manual_seed(42)
    co19_model = model(out_features=2)
    co19_model.load_state_dict(torch.load('saved_model/covid19_detector.pth'))
    co19_model.eval()

    classes = ['covid', 'normal']

    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor()])

    img = Image.open(img)
    img_transformed = transform(img)
    batch_img_transformed = img_transformed.unsqueeze(0)

    pred = co19_model(batch_img_transformed)
    result = classes[pred.argmax(dim=1)]

    prob = torch.nn.Softmax(dim=1)
    prob_value, prob_idx = prob(pred).max(dim=1)

    return result, int(prob_value.item()*100)