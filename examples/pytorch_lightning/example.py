import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pytorch_lightning as pl

from deepview_profile.pl.deepview_callback import DeepViewProfilerCallback

class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def mnist_dataloader(batch_size=32):
    transform = transforms.Compose([transforms.Resize(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    
    mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
    mnist_val = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=batch_size)
    
    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = mnist_dataloader(batch_size=16)
    model = ResNetModel()

    dv_callback = DeepViewProfilerCallback("example")

    trainer = pl.Trainer(
        max_epochs=2, accelerator='gpu', devices=1,
        callbacks=[dv_callback]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)