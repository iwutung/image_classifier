import argparse
import json
from PIL import Image
import torch
from torch import nn
import numpy as np
from torchvision import models, transforms

def arg_parse():
    parser = argparse.ArgumentParser(description='Predict Image Class')

    parser.add_argument('image_dir')
    parser.add_argument('checkpoint_dir')
    parser.add_argument('--gpu', help='Use GPU for training', default='cpu')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes', default=5)
    parser.add_argument('--category_names', help='Use a mapping of categories to real names', default='cat_to_name.json')

    return parser.parse_args()

def load_model(arch):
    exec(f'model = models.{arch}(pretrained=True)', globals())

    for param in model.parameters():
        param.requires_grad = False
    return model

def initialize_classifier(model, hidden_units=4096, out_features=102):
    in_features = model.classifier.in_features
    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, out_features),
                               nn.LogSoftmax(dim=1))
    return classifier

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    output_features = checkpoint['output_features']

    model = load_model(arch)
    if hasattr('model', 'classifier'):
        model.classifier = initialize_classifier(model, hidden_units, output_features)
    else:
        model.fc = initialize_classifier(model, hidden_units, output_features)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    image = Image.open(image_path)
    
    # Resize image where the shortest side is 256,
    # keeping the aspect ratio
    w, h = image.size
    aspect = w/h
    if aspect > 1:
        image = image.resize((int(aspect*256), 256)) 
    else:
        image = image.resize((256, int(256/aspect)))
    
    # crop out the center 224x224 of the image
    w, h = image.size
    centerbox = ((w - 224)/2, (h - 224)/2, (w + 224)/2, (h + 224)/2)
    image = image.crop(centerbox)
    
    # convert to numpy array
    image = np.array(image)/255
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    # Reorder the dimensions
    image = image.transpose(2, 0, 1)

    return image

def predict(image_path, model, topk):

    input_img = process_image(image_path)
    input_img = torch.from_numpy(input_img)
    input_img = input_img.unsqueeze(0)
    input_img = input_img.type(torch.FloatTensor)

    out = model(input_img)

    probs = torch.exp(out)

    top_probs, top_classes = probs.topk(topk)
    
    top_probs = top_probs.numpy()[0]
    top_classes = top_classes.numpy()[0]
    
    # Map classes
    mapping = {v:k for k, v in model.class_to_idx.items()}
    top_classes = [mapping[int(class_)] for class_ in top_classes]

    return top_probs, top_classes

def main():
    args = arg_parse()

    image_dir = args.image_dir
    checkpoint_dir = args.checkpoint_dir
    gpu = args.gpu
    top_k = args.top_k
    category_names = args.category_names

    with open(args.category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)

    model = load_checkpoint(checkpoint_dir)

    image_tensor = process_image(args.image_dir)

    top_probs, top_classes = predict(image_path, model, topk)

    class_names = [cat_to_name[class_] for class_ in top_classes]

    for i in range(top_k):
        print(f'{class_names[i]} with a probability of {top_probs[i]}')

if __name__ == '__main__':
    main()
