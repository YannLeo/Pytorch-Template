from torchvision import datasets, transforms
import torch


def MNISTDataset(path='/home/rxy/Datasets', size=28, channels=3, train=True):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # automatically normalize to [0, 1]
        # transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.repeat(channels, 1, 1)),
    ])

    return datasets.MNIST(path, train=True, download=False, transform=transform) if train else \
        datasets.MNIST(path, train=False, download=False, transform=transform)


if __name__ == '__main__':

    train_loader = torch.utils.data.DataLoader(
        MNISTDataset('/home/rxy/Datasets', channels=3, train=True), batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        MNISTDataset('/home/rxy/Datasets', channels=3, train=False), batch_size=32, shuffle=False)

    for x, y in test_loader:
        print(x.shape, y.shape)
        print(x[0][2] == MNISTDataset('/home/rxy/Datasets', train=False)[0][0])
        break
