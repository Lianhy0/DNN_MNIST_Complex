import matplotlib.pyplot as plt
import numpy as np
import torch

def PlotPicture(x1, x2, x3, x4, x5):
    """
    PlotPicture aims to plot the picture of the parameter.
    x1~x5: 5 of the same kind parameter.
    :return: picture
    """
    if torch.is_tensor(x1):
        x11 = x1.detach().numpy()[0, 0]
        x22 = x2.detach().numpy()[0, 0]
        x33 = x3.detach().numpy()[0, 0]
        x44 = x4.detach().numpy()[0, 0]
        x55 = x5.detach().numpy()[0, 0]
    else:
        x11 = x1
        x22 = x2
        x33 = x3
        x44 = x4
        x55 = x5

    plt.figure(1)
    plt.imshow(x11)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.axis('off')
    plt.savefig('./testpicture0/example1/layer1.png', transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.figure(2)
    plt.imshow(x22)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.axis('off')
    plt.savefig('./testpicture0/example1/layer2.png', transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.figure(3)
    plt.imshow(x33)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.axis('off')
    plt.savefig('./testpicture0/example1/layer3.png', transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.figure(4)
    plt.imshow(x44)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.axis('off')
    plt.savefig('./testpicture0/example1/layer4.png', transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.figure(5)
    plt.imshow(x55)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.axis('off')
    plt.savefig('./testpicture0/example1/layer5.png', transparent=True, bbox_inches='tight', pad_inches=0.0)

    plt.show()



def ComparePredictResult(data, result, xpredict, target):
    """
    Compare the result result and the target.
    :param data: input data         tensor[b, 1, w, h]
    :param predict: result iamge
    :param target: true class
    """
    datanumpy = data.detach().numpy()[0, 0]
    resultnumpy = result.detach().numpy()[0, 0]
    xpredictnumpy = np.argmax(xpredict.detach().numpy())
    targetnumpy = target.detach().numpy()

    print('predict:', xpredictnumpy, 'label:', targetnumpy)

    plt.figure(1)
    plt.imshow(datanumpy)
    plt.axis('off')
    # plt.xticks(fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.savefig('./testpicture0/example2/data.png', transparent=True, bbox_inches='tight', pad_inches=0.0)

    testtool = resultnumpy

    plt.figure(3)
    plt.imshow(testtool)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.axis('off')

    # plt.plot([72, 150, 150, 72, 72], [72, 72, 150, 150, 72], color='white')
    # plt.plot([72, 150, 150, 72, 72], [220, 220, 298, 298, 220], color='white')
    # plt.plot([72, 150, 150, 72, 72], [368, 368, 446, 446, 368], color='white')
    # plt.plot([220, 298, 298, 220, 220], [72, 72, 150, 150, 72], color='white')
    # plt.plot([172, 250, 250, 172, 172], [220, 220, 298, 298, 220], color='white')
    # plt.plot([268, 346, 346, 268, 268], [220, 220, 298, 298, 220], color='white')
    # plt.plot([220, 298, 298, 220, 220], [368, 368, 446, 446, 368], color='white')
    # plt.plot([368, 446, 446, 368, 368], [72, 72, 150, 150, 72], color='white')
    # plt.plot([368, 446, 446, 368, 368], [220, 220, 298, 298, 220], color='white')
    # plt.plot([368, 446, 446, 368, 368], [368, 368, 446, 446, 368], color='white')

    # plt.plot([208.5, 229.5, 229.5, 208.5, 208.5], [208.5, 208.5, 229.5, 229.5, 208.5], color='white')
    # plt.plot([208.5, 229.5, 229.5, 208.5, 208.5], [248.5, 248.5, 269.5, 269.5, 248.5], color='white')
    # plt.plot([208.5, 229.5, 229.5, 208.5, 208.5], [288.5, 288.5, 309.5, 309.5, 288.5], color='white')
    # plt.plot([248.5, 269.5, 269.5, 248.5, 248.5], [208.5, 208.5, 229.5, 229.5, 208.5], color='white')
    # plt.plot([235.5, 256.5, 256.5, 235.5, 235.5], [248.5, 248.5, 269.5, 269.5, 248.5], color='white')
    # plt.plot([261.5, 282.5, 282.5, 261.5, 261.5], [248.5, 248.5, 269.5, 269.5, 248.5], color='white')
    # plt.plot([248.5, 269.5, 269.5, 248.5, 248.5], [288.5, 288.5, 309.5, 309.5, 288.5], color='white')
    # plt.plot([288.5, 309.5, 309.5, 288.5, 288.5], [208.5, 208.5, 229.5, 229.5, 208.5], color='white')
    # plt.plot([288.5, 309.5, 309.5, 288.5, 288.5], [248.5, 248.5, 269.5, 269.5, 248.5], color='white')
    # plt.plot([288.5, 309.5, 309.5, 288.5, 288.5], [288.5, 288.5, 309.5, 309.5, 288.5], color='white')

    plt.plot([98.5, 179.5, 179.5, 98.5, 98.5], [98.5, 98.5, 179.5, 179.5, 98.5], color='white')
    plt.plot([83.5, 164.5, 164.5, 83.5, 83.5], [218.5, 218.5, 299.5, 299.5, 218.5], color='white')
    plt.plot([98.5, 179.5, 179.5, 98.5, 98.5], [338.5, 338.5, 419.5, 419.5, 338.5], color='white')

    plt.plot([218.5, 299.5, 299.5, 218.5, 218.5], [98.5, 98.5, 179.5, 179.5, 98.5], color='white')
    plt.plot([173.5, 254.5, 254.5, 173.5, 173.5], [218.5, 218.5, 299.5, 299.5, 218.5], color='white')
    plt.plot([263.5, 344.5, 344.5, 263.5, 263.5], [218.5, 218.5, 299.5, 299.5, 218.5], color='white')
    plt.plot([218.5, 299.5, 299.5, 218.5, 218.5], [338.5, 338.5, 419.5, 419.5, 338.5], color='white')

    plt.plot([338.5, 419.5, 419.5, 338.5, 338.5], [98.5, 98.5, 179.5, 179.5, 98.5], color='white')
    plt.plot([353.5, 434.5, 434.5, 353.5, 353.5], [218.5, 218.5, 299.5, 299.5, 218.5], color='white')
    plt.plot([338.5, 419.5, 419.5, 338.5, 338.5], [338.5, 338.5, 419.5, 419.5, 338.5], color='white')


    # plt.colorbar()
    # plt.savefig('./testpicture0/example2/result.png', transparent=True, bbox_inches='tight', pad_inches=0.0)

    plt.show()

