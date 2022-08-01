from PIL import ImageFile
from torch.utils.data import DataLoader

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device):
    # Setting the module in evaluating mode
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs, dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
    
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        )
    )

def train(model, train_loader, criterion, optimizer, device):
    # Setting the module in training mode
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Setting the gradients to zero
        optimizer.zero_grad()
        
        # Back propagation
        loss.backward()
        
        # Gardient descent
        optimizer.step()
        
        _, preds = torch.max(outputs, dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        running_samples += len(inputs)
        
        if running_samples % 500  == 0:
            accuracy = running_corrects/running_samples
            logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                running_samples,
                len(train_loader.dataset),
                100.0 * (running_samples / len(train_loader.dataset)),
                loss.item(),
                running_corrects,
                running_samples,
                100.0*accuracy)
            )
    
    epoch_loss = running_loss / running_samples
    epoch_acc = running_corrects / running_samples
    
    return model
    
def net():
    # We use pretrained Inception v3 model
    model = models.inception_v3(aux_logits=False, pretrained=True)
    
    # Freezing all convolutional layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Number of inputs for Fully Connected (FC) layer
    num_features = model.fc.in_features
    
    # Adding Fully Connected layers with 'num_features' inputs and
    # 133 outputs, because we have 133 classes (dog breeds)
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    
    return model

def create_data_loaders(data, batch_size):
    logger.info("Get data loaders")
    # Data preprocess for data loaders
    train_preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # Horizontally flip a image with probability 50%
        transforms.Resize(299), # for Inception V3 image must be square with sides of 299px 
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_preprocess = transforms.Compose([
        transforms.Resize(299), # for Inception V3 image must be square with sides of 299px 
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_data_path = os.path.join(data, "train")
    valid_data_path = os.path.join(data, "valid")
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_preprocess)
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_data_path, transform=test_preprocess)
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_data_loader, valid_data_loader

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initializing a model by calling the net function
    model = net()
    model = model.to(device)
    
    # Creating loss criterion
    loss_criterion = nn.CrossEntropyLoss()
    
    # Creating Adam optimizer
    optimizer = optim.Adam(params=model.fc.parameters(), lr=args.lr)
    
    # Creating train data loader
    train_data_loader, valid_data_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch: {}".format(epoch))
        model = train(model, train_data_loader, loss_criterion, optimizer, device)
        test(model, valid_data_loader, loss_criterion, device)
    
    # Saving the trained model
    save_model(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)"
    )

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    
    args=parser.parse_args()
    
    main(args)