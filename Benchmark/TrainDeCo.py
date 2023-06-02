import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from src.Model import Model, DeCoModel
from utils.Config import get_config_from_json
from utils.DataLoader import DataLoader, DataLoader_npy
# from visdom import Visdom
from src.AccuracyFunction import AccuracyFunction
from utils.RegularizationLoss import MyLoss, MyLabelLoss
from utils.ImagePlot import ComparePredictResult
import matplotlib.pyplot as plt

def TrainFunction(config, Model, TrainLoader, TestLoader):
    optimizer = optim.Adam(Model.parameters(), lr=config.learning_rate)
    Accuracy = []
    Loss = []
    Lossratio = []
    Accinitial = 0
    for epoch in range(config.num_epochs):
        Lossepoch = 0
        print(epoch+1)
        for index, (data, target) in enumerate(TrainLoader):
            data = data.to(config.device)
            target = target.to(config.device)
            optimizer.zero_grad()
            output = Model(config, data, 'train', 'true')
            loss = MyLabelLoss(config, output, target)
            loss.backward()
            optimizer.step()
            print(index, '     ', loss)
            Lossepoch = Lossepoch + loss

        Lossepochnumpy = str(Lossepoch.cpu().detach().numpy())
        Loss.append(Lossepochnumpy)

        Acc = AccuracyFunction(config, Model, TestLoader, state='accu')
        Accuracy.append(Acc)

        if Acc >= Accinitial:
            torch.save(Model.state_dict(), "net_best.ckpt")
            param_dict = Model.state_dict()
            for name, param in Model.named_parameters():
                np.set_printoptions(threshold=np.inf)
                A = param.detach().cpu().numpy().tolist()
                param_dict[name] = A
                print(name, ' ', A)

            param_dict['Loss'] = Loss
            param_dict['Acc'] = Accuracy
            param_dict['epoch'] = epoch
            param_dict['Lossratio'] = Lossratio
            with open('./net_best.json', 'w') as Parameters:
                json.dump(param_dict, Parameters, indent=2)

        Accinitial = Acc



if __name__ == '__main__':
    config_file = open('../Configuration/Configuration.json')
    config = get_config_from_json(config_file)

# 训练步骤
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        print("GPU is enabled")
    else:
        config.device = torch.device("cpu")
        print("GPU is not enabled")

# 数据
    TrainLoader, TestLoader = DataLoader_npy(config, True, False)

# 模型
    Model = DeCoModel(config).to(device=config.device)

    # Acc = AccuracyFunction(config, Model, TestLoader, state='accu')

    TrainFunction(config, Model, TrainLoader, TestLoader)