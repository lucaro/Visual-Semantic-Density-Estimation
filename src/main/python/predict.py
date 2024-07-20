import PIL

import torch
from torch import nn
import torchvision.transforms as T

from timm import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

base_model_name = "convnext_tiny_in22k"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = create_model(base_model_name, pretrained=True).to(device)

children = list(base_model.children())
relevant = children[:-1].copy()
relevant.append(children[-1][:-2])
relevant.append(nn.Linear(768, 2048, bias = True))
relevant.append(nn.GELU())
relevant.append(nn.Linear(2048, 1, bias = True))

model = torch.nn.Sequential(*relevant).to(device)
state = torch.load('data/model/checkpoint_best.pt', map_location=torch.device('cpu'))

model.load_state_dict(state['state_dict'])


NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
SIZE = 256

transforms = [
    T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
]
transforms = T.Compose(transforms)

img = PIL.Image.open('/path/to/image.jpg')
img_tensor = transforms(img).unsqueeze(0)

with torch.no_grad():
    prediction = model(img_tensor).cpu().numpy()[0][0]

print(prediction)