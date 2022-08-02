import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

JPEG_CONTENT_TYPE = 'image/jpeg'

def net():
    model = models.inception_v3(aux_logits=False, pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model

def model_fn(model_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = net().to(device)
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(f"Requested an unsupported Content-Type: {content_type}")

def predict_fn(input_data, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    test_transform =  transforms.Compose([
        transforms.Resize(299), # for Inception V3 image must be square with sides of 299px 
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transformed_input_data = test_transform(input_data)
    
    if torch.cuda.is_available():
        # put data into GPU
        transformed_input_data = transformed_input_data.cuda()
        
    model.eval()
    with torch.no_grad():
        prediction = model(transformed_input_data.unsqueeze(0))
        
    return prediction