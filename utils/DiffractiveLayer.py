import numpy as np
import torch
from utils.Config import get_config_from_json


class DiffractiveLayer(torch.nn.Module):
    """
    Diffractive layer of the model.
    """
    def __init__(self, config):
        super(DiffractiveLayer, self).__init__()
        self.size = config.SLM_padding_pixel_num
        self.dL = self.size * config.SLM_pixel_size
        self.df = 1.0 / self.dL
        self.dist = config.diffractive_distance
        self.lmb = config.wave_length
        self.k = np.pi * 2.0 / self.lmb
        self.H_z = torch.from_numpy(self.InitH())
        self.H_z = self.H_z.to(config.device)
        self.H_z_r = self.H_z.real
        self.H_z_i = self.H_z.imag

    def InitH(self):
        N = self.size
        df = self.df
        k = self.k
        d = self.dist
        lmb = self.lmb
        def phase(i, j):
            i -= N // 2
            j -= N // 2
            return ((i * df) * (i * df) + (j * df) * (j * df))

        ph = np.fromfunction(phase, shape=(N, N), dtype=np.float32)
        H = np.exp(1.0j * k * d) * np.exp(-1.0j * lmb * np.pi * d * ph)
        return H

    def forward(self, x):
        x11 = torch.fft.fftshift(torch.fft.fft2(x))
        x22 = x11 * self.H_z
        x33 = torch.fft.ifft2(torch.fft.ifftshift(x22))
        xampp = x33.real * x33.real + x33.imag * x33.imag
        return xampp



if __name__ == '__main__':
    config_file = open('../Configuration/Configuration.json')
    config = get_config_from_json(config_file)
    Net = DiffractiveLayer(config)

    amp = np.load('../7.npy')
    import torchvision.transforms as trans
    padtool1 = trans.Pad(400, 0, 'constant')
    amp_tensor = padtool1(torch.from_numpy(amp).unsqueeze(0))

    # 复振幅
    c0Amp = torch.from_numpy(np.random.random(size=[400, 400]))
    c0Angle = torch.from_numpy(np.random.random(size=[400, 400]) * np.pi * 2)
    complex0 = padtool1(c0Amp * torch.exp(1.0j * c0Angle)).unsqueeze(0)

    # c0Real = torch.from_numpy(0.7071 * (np.random.random(size=[400, 400]) - 0.5) * 2)
    # c0Imag = torch.from_numpy(0.7071 * (np.random.random(size=[400, 400]) - 0.5) * 2)
    # complex0 = padtool1(c0Real + 1.0j * c0Imag).unsqueeze(0)

    # 衍射
    diffRe = Net(amp_tensor * complex0)

    # Result
    import matplotlib.pyplot as plt
    diffReAbs = abs(diffRe) ** 2
    c0Angle = np.angle(complex0)
    plt.imshow(diffReAbs[0, 400:800, 400:800])
    plt.show()

    plt.imshow(c0Angle[0, 400:800, 400:800], vmin=-1 * np.pi, vmax=np.pi)
    plt.show()

    print(0)


