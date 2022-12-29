import torch

from gui.booth import Booth
from model.models.model import MultimodalModelTFusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultimodalModelTFusion()
model.load_state_dict(torch.load("./weights/model_76_final.pth", map_location=device))
model.to(device)

b = Booth(model=model, device=device, plot_audio=False)
