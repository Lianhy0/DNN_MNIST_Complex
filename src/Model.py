import torch
import torch.nn
import numpy as np
import torch.nn.functional as F
from utils.MiddleLayer import MiddleLayer, LastLayer
from utils.BinarizationLayer import BinarizationLayer, UniformLayer
from torchvision.transforms import transforms
from src.DetectRegion import DetectRegion

class Model(torch.nn.Module):
    """
    Model used in the project.
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.phase_size = config.SLM_pixel_num
        self.pi2 = torch.tensor(np.pi * 2)
        self.CAM2DMD = transforms.Resize(int(config.DMD_pixel_num))

        # self.real1 = torch.nn.Parameter(
        #     torch.from_numpy(0.7071 * (np.random.random(size=[self.phase_size, self.phase_size]) - 0.5) * 2),
        #     requires_grad=True)
        # self.imag1 = torch.nn.Parameter(
        #     torch.from_numpy(0.7071 * (np.random.random(size=[self.phase_size, self.phase_size]) - 0.5) * 2),
        #     requires_grad=True)

        # self.real2 = torch.nn.Parameter(
        #     torch.from_numpy(0.7071 * (np.random.random(size=[self.phase_size, self.phase_size]) - 0.5) * 2),
        #     requires_grad=True)
        # self.imag2 = torch.nn.Parameter(
        #     torch.from_numpy(0.7071 * (np.random.random(size=[self.phase_size, self.phase_size]) - 0.5) * 2),
        #     requires_grad=True)

        self.real3 = torch.nn.Parameter(
            torch.from_numpy(0.7071 * (np.random.random(size=[self.phase_size, self.phase_size]) - 0.5) * 2),
            requires_grad=True)
        self.imag3 = torch.nn.Parameter(
            torch.from_numpy(0.7071 * (np.random.random(size=[self.phase_size, self.phase_size]) - 0.5) * 2),
            requires_grad=True)

        # self.alpha1 = torch.nn.Parameter(torch.from_numpy(1200 * (np.random.random([1, 1]) + 1)), requires_grad=True)
        # self.beta1 = torch.nn.Parameter(torch.from_numpy(np.ones([1, 1])), requires_grad=True)

        self.alpha2 = torch.nn.Parameter(torch.from_numpy(1200 * (np.random.random([1, 1]) + 1)), requires_grad=True)
        self.beta2 = torch.nn.Parameter(torch.from_numpy(np.ones([1, 1])), requires_grad=True)

        self.middleLayer = MiddleLayer(config)
        self.LastLayer = LastLayer(config)

    def forward(self, config, xDMD1, state='train', checkparameter='phase'):
        """
        Model used in the network.
        :param config: configuration
        :param xDMD1: input image    [b, 1, 192, 192] tensor
        :param state: current state   train/accu/check
                train: represent the train process
                accu: represent the accuracy process
                check: represent the check parameter process(aims to plot the image)
                predicteresult: represent the prediction class and the CAM of the last layer
        :param checkparameter: SLM/DMD/CAM/phase
        :return:
        """
        # complex0 = self.real0 + 1.0j * self.imag0
        # complex1 = self.real1 + 1.0j * self.imag1
        # complex2 = self.real2 + 1.0j * self.imag2
        complex3 = self.real3 + 1.0j * self.imag3

        xCAM2 = xDMD1

        # process scatter
        # xCAM0, xDMD0 = self.middleLayer(config, xDMD1, complex0, state)
        # xCAM0 = self.CAM2DMD(xCAM0)
        # xUni0 = UniformLayer(config, xCAM0, state)
        # xBin0 = BinarizationLayer(xUni0, self.alpha0, self.beta0, state, config)

        # process1
        # xCAM1, xDMD1 = self.middleLayer(config, xDMD1, complex1, state)
        # xCAM1 = self.CAM2DMD(xCAM1)
        # xUni1 = UniformLayer(config, xCAM1, state)
        # xBin1 = BinarizationLayer(xUni1, self.alpha1, self.beta1, state, config)

        # process2
        # xCAM2, xDMD2 = self.middleLayer(config, xBin1, complex2, state)
        xCAM2 = self.CAM2DMD(xCAM2)
        xUni2 = UniformLayer(config, xCAM2, state)
        xBin2 = BinarizationLayer(xUni2, self.alpha2, self.beta2, state, config)

        # process3
        xresult, xCAM3, xDMD3 = self.LastLayer(config, xBin2, complex3, state)


        if state == 'train':
            # return xresult, phase0, phase1, phase2, phase3
            return xresult
        elif state == 'accu':
            # return F.log_softmax(xresult, dim=1)
            return xresult
        elif state == 'check':
            if checkparameter == 'CAM':
                return xCAM3
                # return xCAM0, xCAM1, xCAM2, xCAM3
            # elif checkparameter == 'SLM':
            #     return xSLM3
                # return xSLM0, xSLM1, xSLM2, xSLM3
            elif checkparameter == 'DMD':
                return xDMD3
                # return xDMD0, xDMD1, xDMD2, xDMD3
            # elif checkparameter == 'phase':
            #     return phase3
            elif checkparameter == 'complex':
                return complex3
                # return phase0, phase1, phase2, phase3
        elif state == 'predictresult':
            return F.log_softmax(xresult, dim=1), xCAM3



class DeCoModel(torch.nn.Module):
    def __init__(self, config):
        super(DeCoModel, self).__init__()
        self.decisionCoeff = torch.nn.Parameter(torch.from_numpy(np.ones([config.nClass])), requires_grad=True)

    def initDeciMask(self, config):
        self.decisionMask = torch.from_numpy(np.ones([config.CAM_pixel_num, config.CAM_pixel_num]))
        self.decisionMask[99:179, 99:179] = self.decisionCoeff[0]
        self.decisionMask[99:179, 219:299] = self.decisionCoeff[1]
        self.decisionMask[99:179, 339:419] = self.decisionCoeff[2]
        self.decisionMask[219:299, 84:164] = self.decisionCoeff[3]
        self.decisionMask[219:299, 174:254] = self.decisionCoeff[4]
        self.decisionMask[219:299, 264:344] = self.decisionCoeff[5]
        self.decisionMask[219:299, 354:434] = self.decisionCoeff[6]
        self.decisionMask[339:419, 99:179] = self.decisionCoeff[7]
        self.decisionMask[339:419, 219:299] = self.decisionCoeff[8]
        self.decisionMask[339:419, 339:419] = self.decisionCoeff[9]
        return self.decisionMask

    def forward(self, config, x, state, checkparameter='true'):
        self.decisionMask = self.initDeciMask(config).to(config.device)
        xDeMask = self.decisionMask * x
        if state == 'train':
            return xDeMask
        elif state == 'accu':
            return DetectRegion(xDeMask)