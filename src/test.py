from utils import create_model, predict_image
import torch
import os

if __name__ == "__main__":
    model_path = "models/resnet50.pth"
    image_path = input("Введите путь до подававаемого изображения: ")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    class_names = ["Fracture", "Not Fracture"]

    predicted_class = predict_image(model, image_path, class_names)

    print(f"The predicted class is: {predicted_class}")
