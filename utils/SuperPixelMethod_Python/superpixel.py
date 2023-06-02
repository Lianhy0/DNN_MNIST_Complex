from numpy import loadtxt, zeros, meshgrid, ceil, floor, arange, exp, pi, angle, real, imag, zeros_like, roll, concatenate, conj, square, sqrt, max, round, sum, mod, where, insert, append
from numpy.linalg import norm
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from tqdm import trange
import torch

class SuperPixelMethod(object):
    _lookupTable = None
    _targetFields = None
    _maxAmplitude = None
    _stepSize = None
    _lookupTable_x0 = None

    def __init__(self, config, resolution=8, superpixelSize=4,  normalization="maxAmplitude"):
        self._resolution = resolution
        self._superpixelSize = superpixelSize
        self._normalization = normalization # "maxAmplitude" or "highRes"
        # self._createLookupTable()
        if (SuperPixelMethod._maxAmplitude is None) or (SuperPixelMethod._stepSize is None):
            gridParameters = loadtxt('../utils/SuperPixelMethod_Python/gridParameters.csv', delimiter=',')
            SuperPixelMethod._maxAmplitude = gridParameters[0]
            SuperPixelMethod._stepSize = gridParameters[1]
        if SuperPixelMethod._lookupTable is None:
            SuperPixelMethod._lookupTable = loadtxt('../utils/SuperPixelMethod_Python/lookupTable.csv', delimiter=',')
            SuperPixelMethod._lookupTable_x0 = (len(SuperPixelMethod._lookupTable) + 1)/2
        if SuperPixelMethod._targetFields is None:
            SuperPixelMethod._targetFields = loadtxt('../utils/SuperPixelMethod_Python/targetFields.csv', dtype=complex, converters={0: lambda x: complex(x.decode().replace('i', 'j'))}, delimiter=',')
        self.config = config

    def _createLookupTable(self):
        if (SuperPixelMethod._lookupTable is None) or (SuperPixelMethod._targetFields is None):
            SuperPixelMethod._targetFields, SuperPixelMethod._lookupTable, SuperPixelMethod._maxAmplitude, SuperPixelMethod._stepSize = _createDMDsuperpixelLookupTable(self._superpixelSize)
            SuperPixelMethod._lookupTable_x0 = (len(SuperPixelMethod._lookupTable) + 1)/2

    def _fourierMask(self):
        self._maskCenterX = int(ceil((self._nx - 1) / 2))
        self._maskCenterY = int(ceil((self._ny - 1) / 2))
        self._xx, self._yy = meshgrid(arange(self._nx), arange(self._ny))
        self._mask = ((self._yy - self._maskCenterY) ** 2 + (self._ny / self._nx * (self._xx - self._maskCenterX)) ** 2 < (self._ny / self._resolution / 2) ** 2).astype('float')
                                                                       

    def _rescaleTargetToSuperpixelResolution(self):
        self._nSuperpixelX = self._nx // self._superpixelSize
        self._nSuperpixelY = self._ny // self._superpixelSize
        # Create mask
        FourierMaskSuperpixelResolution = torch.from_numpy(((self._yy - self._maskCenterY) ** 2 + (self._ny / self._nx * (self._xx - self._maskCenterX)) ** 2 < (self._ny / self._superpixelSize / 2) ** 2).astype('float')).to(self.config.device)
        # E_target_ft = fftshift(fft2(ifftshift(self.E_target)))
        E_target_ft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(self.E_target)))
        # Apply mask
        E_target_ft = FourierMaskSuperpixelResolution * E_target_ft
        # Remove zeros outside of mask
        E_superpixelResolution_ft = E_target_ft[int(self._maskCenterY - ceil((self._nSuperpixelY - 1) / 2)) : int(self._maskCenterY + floor((self._nSuperpixelY + 1) / 2)), int(self._maskCenterX - ceil((self._nSuperpixelX - 1) / 2)) : int(self._maskCenterX + floor((self._nSuperpixelX + 1) / 2))]
        # Add phase gradient to compensate for anomalous 1.5 pixel shift in real plane
        tmpX = arange(1, self._nSuperpixelX + 1) / self._nSuperpixelX
        tmpY = arange(1, self._nSuperpixelY + 1) / self._nSuperpixelY
        phaseFactor = torch.from_numpy(exp((tmpY.reshape(self._nSuperpixelY, 1) + tmpX.reshape(1, self._nSuperpixelX)) * pi * 1j * 3/4)).to(self.config.device)
        E_superpixelResolution_ft = E_superpixelResolution_ft * phaseFactor
        # Fourier transform back to DMD plane
        self._E_superpixelResolution = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(E_superpixelResolution_ft)))

    def _setMin(self, arr):
        arr[arr > 1] = 1
        return arr

    def _phaseAndAmplitude_to_DMDpixels_lookupTable(self):
        # Decrease maximum amplitude to 1 if needed
        self._E_superpixelResolution = torch.exp(torch.angle(self._E_superpixelResolution) * 1j) * self._setMin(abs(self._E_superpixelResolution))
        # Correct for overall phase offset
        self._E_superpixelResolution = self._E_superpixelResolution * exp(11/16 * pi * 1j)
        # Choose normalization: maxAmplitude for highest efficiency, highRes to restrict the modulation to a smaller and denser disk in the complex plane
        if self._normalization == 'maxAmplitude':
            self._E_superpixelResolution = self._E_superpixelResolution * SuperPixelMethod._maxAmplitude
        elif self._normalization == 'highRes':
            self._E_superpixelResolution = self._E_superpixelResolution * 0.906131
        else:
            raise NameError('Normalization method input error!')
        # Loop over superpixels, Find correct combination of pixels to turn on in the lookup table and put them into the 'DMDpixels' matrix that contains the DMD pattern
        self._E_superpixelResolution = self._E_superpixelResolution / SuperPixelMethod._stepSize

        # idx = (SuperPixelMethod._lookupTable[round(imag(self._E_superpixelResolution) + torch.tensor(SuperPixelMethod._lookupTable_x0 - 1)).astype('int'), round(real(self._E_superpixelResolution) + torch.tensor(SuperPixelMethod._lookupTable_x0 - 1)).astype('int')]).astype('int')
        idx = SuperPixelMethod._lookupTable[round(imag(self._E_superpixelResolution) + torch.tensor(SuperPixelMethod._lookupTable_x0 - 1)), torch.round(real(self._E_superpixelResolution) + torch.tensor(SuperPixelMethod._lookupTable_x0 - 1))]
        pixel = real(SuperPixelMethod._targetFields[idx - 1, 1 : torch.from_numpy(self._superpixelSize) ** 2 + 1]).astype('int')
        pixels = zeros_like(pixel)
        for i in range(self._superpixelSize):
            pixels[i::self._superpixelSize, :, :] = roll(pixel[i::self._superpixelSize, :, :], -i*self._superpixelSize, axis=2)
        
        DMDpixels = concatenate(concatenate(pixels.reshape(self._nSuperpixelY, self._nSuperpixelX, self._superpixelSize, self._superpixelSize, order='F'), axis=1), axis=1)
        phaseFactor = exp((arange(1, self._ny + 1).reshape(self._ny, 1) + 4 * arange(1, self._nx + 1).reshape(1, self._nx)) * pi * 1j / 8)
        DMDpixels = DMDpixels * phaseFactor

        return DMDpixels

    def _calculateFidelity(self, E1, E2):
        gamma = sum(E1 * conj(E2)) / (norm(E1) * norm(E2))
        fidelity = abs(gamma) ** 2

        return fidelity

    def _quantification(self, DMDpixels):
        # First lens
        DMDpixels_ft = fftshift(fft2(ifftshift(DMDpixels)))
        # Spatial filter
        DMDpixels_ft = DMDpixels_ft * self._mask
        # Second lens
        E_obtained = fftshift(ifft2(ifftshift(DMDpixels_ft)))
        # Intensity efficiency ( total intensity in E_obtained / total incident intensity)
        efficiency = sum(square(abs(E_obtained/sqrt(self._nx * self._ny))))
        # Normalize fields
        E_obtained = E_obtained / norm(E_obtained)
        self.E_target = self.E_target / norm(self.E_target)
        # Calculate the fidelity
        fidelity = self._calculateFidelity(self.E_target, E_obtained)

        return E_obtained, fidelity, efficiency

    def encoding(self, E_target, isQuantification=False):
        self.E_target = E_target
        self._ny, self._nx = self.E_target.shape
        # for simplicity, DMD pixels equal to the size of E_target
        # resolution is given in DMDpixels
        # Create Fourier mask
        self._fourierMask()
        # Rescale the target to the superpixel resolution
        self._rescaleTargetToSuperpixelResolution()
        # Normalize such that the maximum amplitude is 1
        self._E_superpixelResolution = 1 * self._E_superpixelResolution / (torch.max(abs(self._E_superpixelResolution)))
        # Calculate which DMD pixels to turn on according to the superpixel method
        DMDpixels = self._phaseAndAmplitude_to_DMDpixels_lookupTable()
        DMDpattern = abs(DMDpixels)
        # Quantification
        if isQuantification:
            E_obtained, fidelity, efficiency = self._quantification(DMDpixels)
            return DMDpattern, E_obtained, fidelity, efficiency

        return DMDpattern
            
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    TanAlfa = 0.2
    TanBeta = 0.2
    M_Pixels = 1600
    N_Pixels = 1600
    resolution = 8

    Cor_x = np.arange(- (N_Pixels // 2), (N_Pixels // 2))
    Cor_y = np.arange(- (M_Pixels // 2), (M_Pixels // 2))
    Cor_X, Cor_Y = np.meshgrid(Cor_x, Cor_y)

    TheoPhase = TanAlfa * Cor_X + TanBeta * Cor_Y
    EAbs = np.ones([M_Pixels, N_Pixels])

    TheoField = EAbs * np.exp(1j * TheoPhase)

    # startTime = time.time()
    sp = SuperPixelMethod()

    # createLookupTableTime = time.time()
    # print(createLookupTableTime - startTime)

    DMDpattern_superpixel, E_superpixel, fidelity_superpixel, efficiency_superpixel = sp.encoding(TheoField, isQuantification=True)
    # DMDpattern_superpixel = sp.encoding(TheoField)
    # print(time.time()-createLookupTableTime)
    print(fidelity_superpixel, efficiency_superpixel)
    plt.imshow(abs(E_superpixel)**2)
    plt.show()

    print(0)