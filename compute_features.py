import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import clip  # Added CLIP import

# data_dir = '/hd_data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
# image_dir = os.path.join(data_dir, 'JPEGImages')
# val_file = 'data/voc_val.txt'
# data_dir = '/hd_data/Paris/'
# image_dir = os.path.join(data_dir, 'paris')
# val_file = 'data/val_paris.txt'
DATASET = 'paris'
MODEL = 'clip'  # Changed to use CLIP by default
data_dir = 'Paris'
image_dir = os.path.join(data_dir, 'images')
list_of_images = os.path.join(data_dir, 'list_of_images.txt')
if __name__ == '__main__':
    #reading data
    with open(list_of_images, "r+") as file: 
        files = [f.split('\t') for f in file]
        
    # check GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model and preprocessing setup
    model = None
    preprocess = None
    if MODEL == 'resnet18':
        model = models.resnet18(pretrained=True).to(device)
        model.fc = torch.nn.Identity()
        dim = 512
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ])
    elif MODEL == 'resnet34':
        model = models.resnet34(pretrained=True).to(device)
        model.fc = torch.nn.Identity()
        dim = 512
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ])
    elif MODEL == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        dim = 384
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ])
    elif MODEL == 'clip':
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        model = clip_model.encode_image
        dim = 512  # CLIP ViT-B/32 output dimension
    else:
        raise ValueError(f"Unknown model: {MODEL}")

    #Pasamos la imagen por el modelo
    with torch.no_grad():        
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype = np.float32)        
        for i, file in enumerate(files):                
            filename = os.path.join(image_dir, file[0])
            image = Image.open(filename).convert('RGB')
            image = preprocess(image).unsqueeze(0).to(device)
            if MODEL == 'clip':
                features[i,:] = model(image).cpu().float()[0,:]
            else:
                features[i,:] = model(image).cpu()[0,:]
            if i%100 == 0:
                print('{}/{}'.format(i, n_images))            
                
        feat_file = os.path.join('data', 'feat_{}_{}.npy'.format(MODEL, DATASET))
        np.save(feat_file, features)
        print('saving data ok')