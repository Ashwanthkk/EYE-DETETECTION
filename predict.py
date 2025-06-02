import torch
import numpy as np
import cv2 as cv
from torchvision import transforms 
from PIL import Image
from neural_network import EyeCnn


def predict_eye(image, weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeCnn().to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model.load_state_dict(torch.load(weight, map_location="cpu"))
    model.eval()

    # Fix starts here 👇
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # Convert np array to PIL Image

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    landmarks = output.view(-1, 2).cpu().numpy()
    return landmarks


def output():
    weig="/content/drive/MyDrive/weights.pth"
    image="/content/07022016176_face_2.jpg"
    image_open=Image.open(image)
    image_cv = cv.cvtColor(np.array(image_open.resize((224, 224))), cv.COLOR_RGB2BGR)

    result=predict_eye(image,weig,)

    right_eye=result[2]
    eye_center = (int(right_eye[0]+7), int(right_eye[1]))
    cv.ellipse(image_cv,center=eye_center, axes=(20, 10),angle=0, startAngle=0,endAngle=360, color=(255, 0, 0),thickness=2 )  # Blue
    cv.imshow(image_cv)