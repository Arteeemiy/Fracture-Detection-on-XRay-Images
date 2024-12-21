import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def load_data(data_dir, batch_size, is_train=True):
    transform = train_transforms if is_train else test_transforms

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=4
    )

    return data_loader

def calculate_metrics(y_true, y_pred):
    accuracy = (y_true == y_pred).sum() / len(y_true)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    return accuracy, precision, recall, f1

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def create_model():
    from torchvision import models

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(
        model.fc.in_features, 2
    )  
    return model

def predict_image(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]
