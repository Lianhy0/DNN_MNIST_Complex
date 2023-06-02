import torch.nn
import torch
import numpy as np
from torchvision.transforms import transforms
from utils.DiffractiveLayer import DiffractiveLayer
from src.DetectRegion import DetectRegion
from utils.BinarizationLayer import UniformLayer, BinarizationLayer

class MiddleLayer(torch.nn.Module):
    """
    LastLayer represent the final layer.
    """
    def __init__(self, config):
        super(MiddleLayer, self).__init__()
        self.DMD2SLM = transforms.Resize(int(config.SLM_pixel_num))
        self.DiffractiveLayer = DiffractiveLayer(config)
        self.SLM2CAM = transforms.Resize(int(config.CAM_pixel_num))
        self.pad = transforms.Pad(padding=int((config.SLM_padding_pixel_num - config.SLM_pixel_num) / 2), fill=0,
                                  padding_mode='constant')
        self.centercrop = transforms.CenterCrop((int(config.DMD_pixel_num), int(config.DMD_pixel_num)))
        self.bg_intensity = torch.from_numpy(np.load('../white_bg_400_norm.npy')).to(config.device)
        self.mask = torch.from_numpy(np.load('../mask.npy')).to(config.device)
        self.CAM2DMD = transforms.Resize(int(config.DMD_pixel_num))

    def forward(self, config, xDMD, complex, state):
        """
        LastLayer represent the last layer of the network.
        :param config: configuration
        :param xDMD: input img of DMD
        :param phase: phase applied in the SLM
        :return:
        """
        xSLM = self.DMD2SLM(xDMD)
        # 滤波
        complex_ft = torch.fft.fftshift(torch.fft.fft2(complex))
        exp_phase = torch.fft.ifft2(torch.fft.ifftshift(complex_ft * self.mask))
        # 滤波结束
        xphase = xSLM * exp_phase
        xphase_bg = xphase * self.bg_intensity
        xphase = self.pad(xphase_bg)
        xlayer = self.DiffractiveLayer(xphase)
        xlayer = self.centercrop(xlayer)
        xCAM = self.SLM2CAM(xlayer)
        return xCAM, xDMD



class LastLayer(torch.nn.Module):
    """
    LastLayer represent the final layer.
    """
    def __init__(self, config):
        super(LastLayer, self).__init__()
        self.DMD2SLM = transforms.Resize(int(config.SLM_pixel_num))
        self.DiffractiveLayer = DiffractiveLayer(config)
        self.SLM2CAM = transforms.Resize(int(config.CAM_pixel_num))
        self.pad = transforms.Pad(padding=int((config.SLM_padding_pixel_num - config.SLM_pixel_num) / 2), fill=0,
                                  padding_mode='constant')
        self.centercrop = transforms.CenterCrop((int(config.DMD_pixel_num), int(config.DMD_pixel_num)))
        self.bg_intensity = torch.from_numpy(np.load('../white_bg_400_norm.npy')).to(config.device)
        self.mask = torch.from_numpy(np.load('../mask.npy')).to(config.device)

    def forward(self, config, xDMD, complex, state):
        """
        LastLayer represent the last layer of the network.
        :param config: configuration
        :param xDMD: input img of DMD
        :param phase: phase applied in the SLM
        :return:
        """
        xSLM = self.DMD2SLM(xDMD)
        # 滤波
        complex_ft = torch.fft.fftshift(torch.fft.fft2(complex))
        exp_phase = torch.fft.ifft2(torch.fft.ifftshift(complex_ft * self.mask))
        # 滤波结束
        xphase = xSLM * exp_phase
        xphase_bg = xphase * self.bg_intensity
        xphase = self.pad(xphase_bg)
        xlayer = self.DiffractiveLayer(xphase)
        xlayer = self.centercrop(xlayer)
        xCAM = self.SLM2CAM(xlayer)
        if state == 'accu':
            xlast = DetectRegion(xCAM)
        elif state == 'predictresult':
            xlast = DetectRegion(xCAM)
        elif state == 'train':
            xlast = UniformLayer(config, xCAM, state)
        else:
            xlast = UniformLayer(config, xCAM, state)

        return xlast, xCAM, xDMD