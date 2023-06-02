import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from utils.Config import get_config_from_json
from utils.MyDataset import MyTrainDataset, MyTestDataset

def DataLoader(config, train_shuffle, test_shuffle):
    """
    Aims to prepare the DataLoader, which include the 2 parts, TrainLoader and TestLoader.
    :param config: Configuration
    :return: TrainLoader and TestLoader
    """
    TrainTrans = transforms.Compose([
        transforms.Resize(config.DMD_pixel_num),
        transforms.ToTensor()
    ])
    TestTrans = transforms.Compose([
        transforms.Resize(config.DMD_pixel_num),
        transforms.ToTensor()
    ])

    config.train_shuffle = train_shuffle
    config.test_shuffle = test_shuffle
    TrainLoader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=TrainTrans),
            batch_size=config.train_batch_size, shuffle=config.train_shuffle)
    TestLoader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=TestTrans),
            batch_size=config.test_batch_size, shuffle=config.test_shuffle)

    return TrainLoader, TestLoader


def DataLoader_npy(config, train_shuffle, test_shuffle):
    TrainTrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.CAM_pixel_num)
    ])
    TestTrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.CAM_pixel_num)
    ])

    my_train_dataset = MyTrainDataset(config, TrainTrans)
    my_test_dataset = MyTestDataset(config, TestTrans)

    config.train_shuffle = train_shuffle
    config.test_shuffle = test_shuffle
    train_loader = torch.utils.data.DataLoader(my_train_dataset, batch_size=config.train_batch_size, shuffle=config.train_shuffle)
    test_loader = torch.utils.data.DataLoader(my_test_dataset, batch_size=config.test_batch_size,
                                               shuffle=config.test_shuffle)

    return train_loader, test_loader



if __name__ == "__main__":
    config_file = open('../Configuration/Configuration.json')
    config = get_config_from_json(config_file)

    TrainLoader, TestLoader = DataLoader(config, True, False)

    import matplotlib.pyplot as plt
    for b, (x, y) in enumerate(TrainLoader):
        x_numpy = x.numpy()[0, 0]
        plt.imshow(x_numpy)
        plt.show()

        print(y)