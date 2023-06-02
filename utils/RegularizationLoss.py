import torch
import torch.nn as nn
import numpy as np

def MyLoss(config, gamma, *args):
    """
    Aims to add the regularization loss to the initial loss.
    :param gamma: weight coefficient of the regularization term
    :param args: several phase parameter
    :return: weighted loss
    """
    LaplaceConv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=0)
    LaplaceConv.weight.data = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float64)
    LaplaceConv.bias.data = torch.zeros(1, dtype=torch.float64)
    LaplaceConv.bias.requires_grad = False
    LaplaceConv.weight.requires_grad = False
    LaplaceConv = LaplaceConv.to(config.device)
    Fnorm_loss = 0
    for ii, phase in enumerate(args):
        phase = phase.unsqueeze(0)
        phase = phase.unsqueeze(0)
        phase_grad2 = LaplaceConv(phase)
        phase_Fnorm = torch.norm(phase_grad2) ** 2
        Fnorm_loss = Fnorm_loss + phase_Fnorm

    gamma_loss = gamma * Fnorm_loss
    return gamma_loss



def MyLabelLoss(config, input, target):
    """
    :param config: configuration
    :param input: batch * 1 * 518 * 518
    :param target: batch tensor
    :return: batch_loss
    """
    criterion = nn.MSELoss()
    label = np.zeros([10, config.CAM_pixel_num, config.CAM_pixel_num])
    for i in range(10):
        label[i] = np.load("../label/%d.npy" % (i))

    label = torch.from_numpy(label).to(config.device)
    batch = input.size(0)
    batch_loss = 0
    for k in range(batch):
        target_num = target[k]
        label_img = label[target_num]
        input_img = input[k, 0]
        loss = criterion(input_img, label_img)
        batch_loss = batch_loss + loss

    return batch_loss




####################################################   验证卷积模块是否正确
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    from Config import get_config_from_json
    import numpy as np
    lenna = cv2.imread('./lenna.jpg', 0)
    laplacian = cv2.Laplacian(lenna, cv2.CV_64F)

    plt.figure('edge')
    plt.imshow(laplacian, cmap='gray')
    # plt.show()

    config_file = 'Configuration/SubBytesconfig.json'
    config = get_config_from_json(config_file)

    lenna = lenna.astype(np.float64)
    lenna_tensor = torch.from_numpy(lenna)
    myloss1 = MyLoss(config, 0, lenna_tensor)

    myloss1numpy = myloss1.numpy()[0, 0]

    plt.figure('myedge')
    plt.imshow(myloss1numpy, cmap='gray')
    # plt.show()

    a = laplacian[1:249, 1:249]
    delta = myloss1numpy + a