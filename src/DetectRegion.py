import torch
import numpy as np
from utils.Config import get_config_from_json
from utils.DataLoader import DataLoader


def DetectRegion(x):                   #十分类
    """
    The function used in the last layer aims to choose the ten DetectRegions corresponding to the 10 classes.
    :param xCAM: CAM Image of the last layer [B, 1, 118, 118]
    :return: 10 * 1 tensor
    """
    # return torch.cat((
    #         x[:, :, 74:148, 74:148].mean(dim=(-1, -2)),
    #         x[:, :, 74:148, 222:296].mean(dim=(-1, -2)),
    #         x[:, :, 74:148, 370:444].mean(dim=(-1, -2)),
    #         x[:, :, 222:296, 74:148].mean(dim=(-1, -2)),
    #         x[:, :, 222:296, 174:248].mean(dim=(-1, -2)),
    #         x[:, :, 222:296, 270:344].mean(dim=(-1, -2)),
    #         x[:, :, 222:296, 370:444].mean(dim=(-1, -2)),
    #         x[:, :, 370:444, 74:148].mean(dim=(-1, -2)),
    #         x[:, :, 370:444, 222:296].mean(dim=(-1, -2)),
    #         x[:, :, 370:444, 370:444].mean(dim=(-1, -2))), dim=-1)

    # return


    # return torch.cat((
    #         x[:, :, 209:229, 209:229].mean(dim=(-1, -2)),
    #         x[:, :, 209:229, 249:269].mean(dim=(-1, -2)),
    #         x[:, :, 209:229, 289:309].mean(dim=(-1, -2)),
    #         x[:, :, 249:269, 209:229].mean(dim=(-1, -2)),
    #         x[:, :, 249:269, 236:256].mean(dim=(-1, -2)),
    #         x[:, :, 249:269, 262:282].mean(dim=(-1, -2)),
    #         x[:, :, 249:269, 289:309].mean(dim=(-1, -2)),
    #         x[:, :, 289:309, 209:229].mean(dim=(-1, -2)),
    #         x[:, :, 289:309, 249:269].mean(dim=(-1, -2)),
    #         x[:, :, 289:309, 289:309].mean(dim=(-1, -2))), dim=-1)

    return torch.cat((
            x[:, :, 109:169, 109:169].mean(dim=(-1, -2)),
            x[:, :, 109:169, 229:289].mean(dim=(-1, -2)),
            x[:, :, 109:169, 349:409].mean(dim=(-1, -2)),
            x[:, :, 229:289, 94:154].mean(dim=(-1, -2)),
            x[:, :, 229:289, 184:244].mean(dim=(-1, -2)),
            x[:, :, 229:289, 274:334].mean(dim=(-1, -2)),
            x[:, :, 229:289, 364:424].mean(dim=(-1, -2)),
            x[:, :, 349:409, 109:169].mean(dim=(-1, -2)),
            x[:, :, 349:409, 229:289].mean(dim=(-1, -2)),
            x[:, :, 349:409, 349:409].mean(dim=(-1, -2))), dim=-1)

#act
    # return torch.cat((
    #     x[:, :, 99:179, 99:179].mean(dim=(-1, -2)),
    #     x[:, :, 99:179, 219:299].mean(dim=(-1, -2)),
    #     x[:, :, 99:179, 339:419].mean(dim=(-1, -2)),
    #     x[:, :, 219:299, 84:164].mean(dim=(-1, -2)),
    #     x[:, :, 219:299, 174:254].mean(dim=(-1, -2)),
    #     x[:, :, 219:299, 264:344].mean(dim=(-1, -2)),
    #     x[:, :, 219:299, 354:434].mean(dim=(-1, -2)),
    #     x[:, :, 339:419, 99:179].mean(dim=(-1, -2)),
    #     x[:, :, 339:419, 219:299].mean(dim=(-1, -2)),
    #     x[:, :, 339:419, 339:419].mean(dim=(-1, -2))), dim=-1)

    # 单像素远距离
    # return torch.cat((
    #     xCAM[..., 49, 49],
    #     xCAM[..., 49, 58],
    #     xCAM[..., 49, 67],
    #     xCAM[..., 58, 49],
    #     xCAM[..., 58, 55],
    #     xCAM[..., 58, 61],
    #     xCAM[..., 58, 67],
    #     xCAM[..., 67, 49],
    #     xCAM[..., 67, 58],
    #     xCAM[..., 67, 67]), dim=-1)


# def Detect1(x):
    # region0 = torch.flatten(x[:, :, 209:229, 209:229])
    # region1 = torch.flatten(x[:, :, 209:229, 249:269])
    # region2 = torch.flatten(x[:, :, 209:229, 289:309])
    # region3 = torch.flatten(x[:, :, 249:269, 209:229])
    # region4 = torch.flatten(x[:, :, 249:269, 236:256])
    # region5 = torch.flatten(x[:, :, 249:269, 262:282])
    # region6 = torch.flatten(x[:, :, 249:269, 289:309])
    # region7 = torch.flatten(x[:, :, 289:309, 209:229])
    # region8 = torch.flatten(x[:, :, 289:309, 249:269])
    # region9 = torch.flatten(x[:, :, 289:309, 289:309])




if __name__ == "__main__":
    x = torch.from_numpy(np.random.randint(low=0, high=2, size=(128, 1, 518, 518)))
    a = DetectRegion(x)
    print(a)