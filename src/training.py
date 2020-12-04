import torch

import pytorch_lightning as pl

from torchvision import datasets, transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from models import MarsModel

hyper_params = {
    "batch_size": 64,
    "num_epochs": 30,
    "learning_rate": 1e-2,
    "optimizer": "sgd",
    "momentum": 0.9,
    "model": "densenet121",
    "num_classes": 15,
    "pretrained": True,
    "transfer_learning": False,
}

checkpoint = ModelCheckpoint(verbose=True, monitor="val_acc", mode="max")

data_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

ctx_train = datasets.ImageFolder(root="./data/train", transform=data_transform)
train_loader = torch.utils.data.DataLoader(
    ctx_train,
    batch_size=hyper_params["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

ctx_val = datasets.ImageFolder(root="./data/val", transform=data_transform)
val_loader = torch.utils.data.DataLoader(
    ctx_val, batch_size=hyper_params["batch_size"], shuffle=True, num_workers=8
)

ctx_test = datasets.ImageFolder(root="./data/test", transform=data_transform)
test_loader = torch.utils.data.DataLoader(
    ctx_test, batch_size=16, shuffle=True, num_workers=4
)

model = MarsModel(hyper_params)

trainer = pl.Trainer(
    gpus=1, max_epochs=hyper_params["num_epochs"], checkpoint_callback=checkpoint
)
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
trainer.test(test_dataloaders=test_loader)
