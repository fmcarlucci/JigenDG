from torchvision import datasets
from torchvision import transforms


def get_dataset(path, mode, image_size):
    if mode == "train":
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1/256., 1/256., 1/256.])  # std=[1/256., 1/256., 1/256.] #[0.229, 0.224, 0.225]
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[1/256., 1/256., 1/256.])  # std=[1/256., 1/256., 1/256.]
        ])
    return datasets.ImageFolder(path, transform=img_transform)
