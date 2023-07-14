from torchvision import datasets, transforms
import torch


def MNISTFedDataset(path='/home/yl/CodeAndData/Data/MNIST/', size=28, channels=3, train=True, id=0, num_clients=3):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # automatically normalize to [0, 1]
        # transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.repeat(channels, 1, 1)),
    ])

    dataset =  datasets.MNIST(path, train=True, download=True, transform=transform) if train else \
        datasets.MNIST(path, train=False, download=True, transform=transform)
    
    idx = torch.arange(len(dataset))
    return torch.utils.data.Subset(dataset, idx[id::num_clients])


if __name__ == '__main__':

    train_loader = torch.utils.data.DataLoader(
        MNISTDataset('/home/public/Datasets', channels=3, train=True), batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        MNISTDataset('/home/public/Datasets', channels=3, train=False), batch_size=32, shuffle=False)

    for x, y in test_loader:
        print(x.shape, y.shape)
        print(x[0][2] == MNISTDataset('/home/public/Datasets', train=False)[0][0])
        break
