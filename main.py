import torch

from gui.booth import Booth
from model.models.model import MultimodalModelTFusion

model = MultimodalModelTFusion()
model.load_state_dict(torch.load("./weights/model_76_final.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

b = Booth(model=model, device=device)
