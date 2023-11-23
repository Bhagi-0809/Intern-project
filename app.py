import os
import torch
from flask import Flask, request, render_template
import torchvision.transforms as transforms
from PIL import Image
app = Flask(__name)
# Load a pre-trained GAN model (e.g., StyleGAN2)
# You can replace this with your own model or use other GAN models.
model=torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub','stylegan2', model_name='paper512')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/generate', methods=['POST'])
def generate_art():
    z = torch.randn(1, 512)  # Random latent vector
    img = model(z)
    # Transform the generated image for display
    img = transforms.ToPILImage()(img[0].detach().cpu())
    img.save("static/generated_art.png")
    return render_template('result.html')
if __name__ == '__main__':
    app.run()
