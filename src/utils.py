import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import precision_score, recall_score, f1_score

# Трансформации для обучения
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

# Трансформации для тестирования
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_data(data_dir, batch_size, is_train=True):
    """
    Загружает данные из директории и возвращает DataLoader для обучения или тестирования.

    :param data_dir: Путь к директории с данными.
    :param batch_size: Размер батча.
    :param is_train: Если True, то применяются трансформации для обучения, иначе для тестирования.
    :return: DataLoader для обучения или тестирования.
    """
    # Выбор трансформации в зависимости от режима
    transform = train_transforms if is_train else test_transforms

    # Загружаем датасет с помощью ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Создаём DataLoader
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=4
    )

    return data_loader


def calculate_metrics(y_true, y_pred):
    """
    Вычисляет и возвращает точность (accuracy), прецизионность (precision), полноту (recall) и F1-меру.

    :param y_true: Список с истинными метками.
    :param y_pred: Список с предсказанными метками.
    :return: Кортеж с метками точности, прецизионности, полноты и F1-меры.
    """
    accuracy = (y_true == y_pred).sum() / len(y_true)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    return accuracy, precision, recall, f1


def save_model(model, path):
    """
    Сохраняет модель в файл.

    :param model: Модель для сохранения.
    :param path: Путь к файлу, куда сохранять модель.
    """
    torch.save(model.state_dict(), path)


def load_model(path, model_class):
    """
    Загружает модель из файла.

    :param path: Путь к файлу с моделью.
    :param model_class: Класс модели для восстановления.
    :return: Модель с загруженными весами.
    """
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def create_model():
    """
    Функция для создания модели. В этом случае используется ResNet50.

    :return: Модель ResNet50.
    """
    from torchvision import models

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(
        model.fc.in_features, 2
    )  # 2 класса (перелом / не перелом)
    return model
