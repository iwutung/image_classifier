import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image

def arg_parse():
    parser = argparse.ArgumentParser(description='Train Classifier')

    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', help='Set directory to save checkpoints', default="checkpoint.pth")
    parser.add_argument('--learning_rate', help='Set the learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='Set the number of hidden units', type=int, default=150)
    parser.add_argument('--output_features', help='Specify the number of output features', type=int, default=102)
    parser.add_argument('--epochs', help='Set the number of epochs', type=int, default=5)
    parser.add_argument('--gpu', help='Use GPU for training', default='cpu')
    parser.add_argument('--arch', help='Choose architecture', default='vgg11')

    return parser.parse_args()

def train_transform(train_dir):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])
    
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    return train_set

def valid_transform(valid_dir):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])
    valid_set = datasets.ImageFolder(valid_dir, transform=transform)
    return valid_set

def train_loader(data, batch_size=64, shuffle=True):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def valid_loader(data, batch_size=64):
    return torch.utils.data.DataLoader(data, batch_size=batch_size)

def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def load_model(arch):
    exec(f'model = models.{arch}(pretrained=True)', globals())

    for param in model.parameters():
        param.requires_grad = False
    return model

def initialize_classifier(model, hidden_units, output_features):
    if hasattr('model', 'classifier'):
        in_features = model.classifier.in_features
    else:
        in_features = model.fc.in_features

    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, output_features),
                               nn.LogSoftmax(dim=1))
    return classifier

def train_model(model, trainloader, validloader, device, optimizer, criterion, epochs=20, print_every=20, steps=0):
    for e in range(epochs):
        training_loss = 0

        model.train()

        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                
                model.eval()

                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        out2 = model(images)
                        loss2 = criterion(out2, labels)
                        valid_loss += loss2.item()

                        probs = torch.exp(out2)
                        top_probs, top_class = probs.topk(1, dim=1)

                        equals = top_class == labels.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {training_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                training_loss = 0
                model.train()
    print('Training Complete')

    return model

def save_checkpoint(model, optimizer, criterion, class_to_idx, path, epochs, arch, hidden_units, output_features):

    model.class_to_idx = class_to_idx
    checkpoint = {'input_size': 224*224*3,
                'output_size': 102,
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict,
                'criterion': criterion,
                'epochs': epochs,
                'arch':arch,
                'class_to_idx': model.class_to_idx,
                'hidden_units': hidden_units,
                'output_features': output_features}
    
    torch.save(checkpoint, path)

def main():
    args = arg_parse()

    data_dir = args.data_dir
    save_path = args.save_dir
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    output_features = args.output_features
    epochs = args.epochs
    gpu = args.gpu
    arch = args.arch

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_set = train_transform(train_dir)
    valid_set = valid_transform(valid_dir)

    trainloader = train_loader(train_set)
    validloader = valid_loader(valid_set)

    if args.gpu:
        device = set_device()

    model = load_model(arch)

    if hasattr('model', 'classifier'):
        model.classifier = initialize_classifier(model, hidden_units, output_features)
    else:
        model.fc = initialize_classifier(model, hidden_units, output_features)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    model.to(device)

    print_every = 10
    steps = 0

    train_model(model, trainloader, validloader, device, optimizer, criterion, epochs, print_every, steps)
    save_checkpoint(model, optimizer, criterion, train_set.class_to_idx, save_path, epochs, arch, hidden_units, output_features)

if __name__ == '__main__':
    main()
