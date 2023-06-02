import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from src.Model import Model
from utils.Config import get_config_from_json
from utils.DataLoader import DataLoader, DataLoader_npy
# from visdom import Visdom
from src.AccuracyFunction import AccuracyFunction
from utils.RegularizationLoss import MyLoss, MyLabelLoss
from utils.ImagePlot import ComparePredictResult
import matplotlib.pyplot as plt


def TrainFunction(config, Model, TrainLoader, TestLoader, Train):
    """
    A function aims to train the model.
    :param config:configuration
    :param Model:D2NN model
    :param TrainLoader: DataLoader used to train the model
    :param TestLoader:DataLoader used to test the model
    :return: Save the trained model parameter.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Model.parameters(), lr=config.learning_rate)
    # vis = Visdom(env='my_loss')
    # line_loss = vis.line(Y=np.array([0]), opts={'xlabel': 'iteration', 'ylabel': 'Loss', 'title': 'Loss'})
    # line_acc = vis.line(Y=np.array([0]), opts={'xlabel': 'epoch', 'ylabel': 'Accuracy', 'title': 'Accuracy'})
    Accuracy = []
    Loss = []
    Lossratio = []
    Accinitial = 0
    for epoch in range(config.num_epochs):
        Lossepoch = 0
        print(epoch+1)
        for index, (data, target) in enumerate(TrainLoader):
            # if Train:
            #     data[data >= 0.5] = 1
            #     data[data < 0.5] = 0
            # print(torch.isnan(data).any())
            data = data.to(config.device)
            target = target.to(config.device)
            # iteration = index + epoch * ((config.train_picture_num // config.train_batch_size) + 1)
            optimizer.zero_grad()
            output = Model(config, data, state='train')
            # init_loss = criterion(output, target)
            init_loss = MyLabelLoss(config, output, target)
            loss = init_loss
            loss.backward()
            optimizer.step()
            # vis.line(X=np.array([iteration]), Y=np.array([loss.detach().numpy()]), win=line_loss, update="append")
            print(index, '     ', loss)
            Lossepoch = Lossepoch + loss

        Lossepochnumpy = str(Lossepoch.cpu().detach().numpy())
        Loss.append(Lossepochnumpy)

        Acc = AccuracyFunction(config, Model, TestLoader, state='accu')
        Accuracy.append(Acc)
        # vis.line(X=np.array([epoch]), Y=np.array([Acc]), win=line_acc, update="append")

        if Acc >= Accinitial:
            torch.save(Model.state_dict(), "net_state_dict_best_3layers_0_ps76miu_60region_complex_mask_imgLoss_adapt2_output.ckpt")
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
            with open('./net_state_dict_best_3layers_0_ps76miu_60region_complex_mask_imgLoss_adapt2_output.json', 'w') as Parameters:
                json.dump(param_dict, Parameters, indent=2)

        Accinitial = Acc


def CheckParameterFunction(config, Model, TestLoader):
    """
    A function aims to check and plot the parameter of the model.
    :param Model: pre-trained model
    :param TestLoader: TestLoader
    :return: plot requiring picture
    """
    for index, (data, target) in enumerate(TestLoader):
        # complex = Model(config, data, state='check', checkparameter='complex')
        import matplotlib.pyplot as plt
        # data = data.to(config.device)
        # target = target.to(config.device)

        # DMD3 = Model(config, data, state='check', checkparameter='DMD')
        # CAM1, CAM2, CAM3 = Model(config, data, state='check', checkparameter='CAM')
        DMD_npy = data.cpu().detach().numpy()[0, 0]
        DMD_npy[DMD_npy > 0.5] = 1
        DMD_npy[DMD_npy < 0.5] = 0
        plt.imshow(DMD_npy)
        plt.show()
        # np.save('C:/Users/yjs/Desktop/MNIST-NoAd/data/amp/%d.npy' % (index), DMD_npy)
        # np.save('C:/Users/yjs/Desktop/MNIST-NoAd/result/layer3/test/sim/%d.npy' % (index), DMD_npy)
        print(index)
        # if index == 200:
        #     break


def CheckPredictionResultFunction(config, Model, TestLoader):
    """
    A function aims to check the prediction result.
    :param config: configuration
    :param Model: pre-trained model
    :param TestLoader: TestLoader
    :return: check and compare the prediction result and the target
    """
    Acc = AccuracyFunction(config, Model, TestLoader, True, state='accu')
    for index, (data, target) in enumerate(TestLoader):
        xpredict, xresult = Model(config, data, state='predictresult')
        ComparePredictResult(data, xresult, xpredict, target)

        print(index)


def ProduceInputTrainImage(config, Model):
    """
    aims to produce the train image for adaptive training.
    """
    config.train_batch_size = 1
    config.train_shuffle = False


    # TrainLoader, _ = DataLoader(config, False, False)
    TrainLoader, TestLoader = DataLoader_npy(config, False, False)


    for index, (data, target) in enumerate(TrainLoader):
        if index == int(config.img_ratio * config.train_picture_num):
            break

        # print(torch.isnan(data).any())
        # if torch.isnan(data).any():
        #     i += 1

        data = data.to(config.device)
        # target = target.to(config.device)
        DMD3 = Model(config, data, state='check', checkparameter='DMD')
        # CAM3 = Model(config, data, state='check', checkparameter='CAM')

        a = DMD3.cpu().detach().numpy()[0, 0]
        a[a > 0.5] = 1
        a[a < 0.5] = 0

        # plt.imshow(a)
        # plt.show()

        # np.save('C:/Users/yjs/Desktop/expData/MNIST-NoSc/result/layer3/train/sim/%d.npy' % (index), a)
        np.save('C:/Users/yjs/Desktop/expData/MNIST-NoSc/data/amp/%d.npy' % (index), a)
        print(index)
        # if index == 200:
        #     break


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

# 记得调整数据集
    TrainLoader, TestLoader = DataLoader(config, True, False)
#     TrainLoader, TestLoader = DataLoader_npy(config, True, False)

# 训练的代码指令
    Model2 = Model(config).to(device=config.device)

# 预加载训练模型（初次训练不需要
#     PATH = './net_state_dict_best_3layers_0_ps76miu_60region_complex_mask_imgLoss_adapt2_input.ckpt'
#     checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
#     Model2.load_state_dict(checkpoint)

    # TrainFunction(config, Model2, TrainLoader, TestLoader, Train=True)

# 其他生成图片的函数需要调整ckpt， 生成图片记得要二值化的图片
#     PATH = './net_state_dict_best_3layers_0_ps76miu_60region_complex_mask_imgLoss_adapt2_output.ckpt'    # 调整训练好的ckpt
#     checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
#     Model2.load_state_dict(checkpoint)

    CheckParameterFunction(config, Model2, TestLoader)
    # CheckPredictionResultFunction(config, Model2, TestLoader)
    # ProduceInputTrainImage(config, Model2)

    print(0)
