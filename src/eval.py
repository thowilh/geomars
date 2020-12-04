import torch

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from torchvision import datasets, transforms
from torch.nn import functional as F

from sklearn.metrics import confusion_matrix

from models import MarsModel
from utils import onehot

network_name = "densenet161"

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
    "pretrained": True,
    "transfer_learning": False,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MarsModel(hyper_params)
checkpoint = torch.load("./models/" + network_name + ".pth")
model.load_state_dict(checkpoint)

ctx_test = datasets.ImageFolder(root="./data/test", transform=data_transform)
test_loader = torch.utils.data.DataLoader(
    ctx_test, batch_size=16, shuffle=True, num_workers=4
)

labels = []
predictions = []
scores = []

# Put on GPU if available
model = model.to(device)

# Set model to eval mode (turns off dropout and moving averages of batchnorm)
model.eval()

# Iterate over test set
with torch.no_grad():
    for i_batch, batch in enumerate(test_loader):
        x, y = batch
        y_hat = model(x.to(device))
        pred = torch.argmax(y_hat, dim=1).cpu()

        labels.append(y.numpy())
        predictions.append(pred.numpy())
        scores.append(F.softmax(y_hat, dim=1).detach().cpu().numpy())

# Computing metrics
labels = np.concatenate(labels, axis=0)
predictions = np.concatenate(predictions, axis=0)
scores = np.concatenate(scores, axis=0)

onehot_labels = [onehot(label, hyper_params["num_classes"]) for label in labels]
onehot_predictions = [
    onehot(prediction, hyper_params["num_classes"]) for prediction in predictions
]

y = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
y_pred = label_binarize(
    predictions, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
)

macro_roc_auc_ovo = roc_auc_score(y, scores, multi_class="ovo", average="macro")

macro_roc_auc_ovr = roc_auc_score(y, scores, multi_class="ovr", average="macro")

acc = metrics.accuracy_score(y, y_pred)

# Writing results to file
print(
    "Classification report for classifier %s:\n%s\n"
    % (network_name, metrics.classification_report(y, y_pred, digits=4)),
    file=open("./results/" + network_name + ".txt", "w"),
)
print(
    "AUROC:\t",
    macro_roc_auc_ovo,
    macro_roc_auc_ovr,
    file=open("./results/" + network_name + ".txt", "a"),
)
print("Acc:\t", acc, file=open("./results/" + network_name + ".txt", "a"))
print("\n", file=open("./results/" + network_name + ".txt", "a"))
print(
    confusion_matrix(labels, predictions),
    file=open("./results/" + network_name + ".txt", "a"),
)
