from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_dataloaders(batch_size: int = 64, val_size: int = 12000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_subset = Subset(train_dataset, range(1000))
    val_subset = Subset(train_dataset, range(1000, 1500))
    test_dataset = Subset(test_dataset, range(500))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
