import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from pathlib import Path
from models import MarsModel
from utils import CTX_Image, download_file
from mrf import MRF

torch.backends.cudnn.benchmark = True

CTX_stripe = "G14_023651_2056_XI_25N148W"
path = "data/raw/" + CTX_stripe + ".tiff"
network_name = "densenet161"

cutouts = {
    "D14_032794_1989_XN_18N282W": (1600, 11000, 7000, 14000),  # Jezero
    "F13_040921_1983_XN_18N024W": (3000, 3000, 5800, 7000),  # Oxia Planum
    "G14_023651_2056_XI_25N148W": (1000, 1000, 2000, 2000),  # Lycus Sulci
}

links = {
    "D14_032794_1989_XN_18N282W": "https://image.mars.asu.edu/stream/D14_032794_1989_XN_18N282W.tiff?image=/mars/images/ctx/mrox_1861/prj_full/D14_032794_1989_XN_18N282W.tiff",  # Jezero
    "F13_040921_1983_XN_18N024W": "https://image.mars.asu.edu/stream/F13_040921_1983_XN_18N024W.tiff?image=/mars/images/ctx/mrox_2375/prj_full/F13_040921_1983_XN_18N024W.tiff",  # Oxia Planum
    "G14_023651_2056_XI_25N148W": "https://image.mars.asu.edu/stream/G14_023651_2056_XI_25N148W.tiff?image=/mars/images/ctx/mrox_1385/prj_full/G14_023651_2056_XI_25N148W.tiff",  # Lycus Sulci
}

if not Path(path).exists():
    # Download file
    print("Dowloading...\n")
    download_file(links[CTX_stripe], "data/raw/")
    print("...Done")
    1


data_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

hyper_params = {
    "batch_size": 64,
    "num_epochs": 15,
    "learning_rate": 1e-2,
    "optimizer": "sgd",
    "momentum": 0.9,
    "model": network_name,
    "num_classes": 15,
    "pretrained": False,
    "transfer_learning": False,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MarsModel(hyper_params)
checkpoint = torch.load("./models/" + network_name + ".pth")
model.load_state_dict(checkpoint)

ctx_image = CTX_Image(path=path, cutout=cutouts[CTX_stripe], transform=data_transform)
test_loader = DataLoader(
    ctx_image, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
)

# Put on GPU if available
model = model.to(device)

# Set model to eval mode (turns off dropout and moving averages of batchnorm)
model.eval()

predictions = []
scores = []
image_pred = []

# Analysing image
with tqdm(test_loader, desc="Testing", leave=False) as t:
    with torch.no_grad():
        for batch in t:
            x, center_pixels = batch
            y_hat = model(x.to(device))

            pred = torch.argmax(y_hat, dim=1).cpu()

            image_pred.append(center_pixels.numpy())
            predictions.append(pred.numpy())
            scores.append(F.softmax(y_hat, dim=1).detach().cpu().numpy())


predictions = np.concatenate(predictions, axis=0)
scores = np.concatenate(scores, axis=0)
scores = np.reshape(
    np.array(scores), ctx_image.out_shape + (int(hyper_params["num_classes"]),)
)
image_pred = np.reshape(np.concatenate(image_pred, axis=0), ctx_image.out_shape)

# Markov random field smoothing
mrf_probabilities = MRF(scores.astype(np.float64))
mrf_classes = np.argmax(mrf_probabilities, axis=2)

# Create Colormap
n = int(hyper_params["num_classes"])
from_list = mpl.colors.LinearSegmentedColormap.from_list
cm = from_list(None, plt.cm.tab20(range(0, n)), n)

# Saving Images
plt.imsave(
    "./results/" + CTX_stripe + "_" + network_name + "_map.png",
    np.reshape(np.array(predictions), ctx_image.out_shape),
    cmap=cm,
    vmin=0,
    vmax=int(hyper_params["num_classes"]),
)
plt.imsave(
    "./results/" + CTX_stripe + "_" + network_name + "_img.png",
    np.dstack(
        [np.reshape(np.concatenate(image_pred, axis=0), ctx_image.out_shape)] * 3
    ),
)
plt.imsave(
    "./results/" + CTX_stripe + "_" + network_name + "_mrf.png",
    mrf_classes,
    cmap=cm,
    vmin=0,
    vmax=int(hyper_params["num_classes"]),
)
