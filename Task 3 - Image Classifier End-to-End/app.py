from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
from model import CustomCNN

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model + class mapping
model = CustomCNN()
ckpt = torch.load('models/cat_dog_model.pth', map_location='cpu')
model.load_state_dict(ckpt["model_state"])
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

if "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
    class_to_idx = ckpt["class_to_idx"]
else:
    model.load_state_dict(ckpt)
    class_to_idx = {"cat": 0, "dog": 1}

idx_to_class = {v: k for k, v in class_to_idx.items()}
model.eval()

# Match training transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename != '':
            fname = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(image_path)
            image_filename = fname

            image = Image.open(image_path).convert('RGB')
            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                idx = int(output.argmax(1).item())
                prediction = idx_to_class[idx].capitalize()

    return render_template('index.html',
                           prediction=prediction,
                           image_filename=image_filename)

if __name__ == '__main__':
    app.run(debug=True)