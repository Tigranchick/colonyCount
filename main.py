import sys
import os
import subprocess
from model import Net
import torch
from PIL import Image
from torchvision import transforms

def train_model():
    # Define the arguments for the generate.py and train.py scripts
    data_path = 'data'
    train_images = '1000'
    val_images = '200'
    network_path = 'model.pth'
    epochs = '5'
    lr = '0.001'
    momentum = '0.9'

    # Run the scripts for training the model
    subprocess.call(["python3", "generate.py", data_path, train_images, val_images])
    subprocess.call(["python3", "train.py", network_path, data_path, epochs, lr, momentum])

def predict_image(image_path, model_path='model.pth'):
    # Load the trained model
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    # Make a prediction
    output = model(image)
    print(f'Predicted number of colonies: {output.item()}')

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-predict':
        if len(sys.argv) != 3:
            print('Usage: python main.py -predict <image_path>')
            sys.exit(1)
        image_path = sys.argv[2]
        predict_image(image_path)
    else:
        train_model()
