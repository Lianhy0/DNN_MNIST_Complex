import torch

def BinarizationLayer(input, alpha, beta, state, config):
    """
    BinarizationLayer aims to package the rest portion of the Model.py except the DiffractiveLayer.
    input: batch, 1, 140, 140, tensor
    alpha: coefficient used in binary process and need to be optimized
    state: represent the current state of the network
    config: configuration
    return: inputuniform: binary result of the layer.batch, 1, 140, 140
    """
    onetool = torch.ones_like(input)
    meantool = torch.ones_like(input)
    meanvalue = input.mean(dim=[-1, -2])
    meanvalue = beta * meanvalue

    if state == 'train':
        batch = input.size(0)
    else:
        batch = config.test_batch_size

    for i in range(batch):
        meantool[i, 0, :, :] = onetool[i, 0, :, :] * meanvalue[i, 0]
    inputaftermean = (input - meantool) * alpha
    inputuniform = torch.sigmoid(inputaftermean)

    return inputuniform

def UniformLayer(config, input, state):
    if state == 'train':
        batch = input.size(0)
    else:
        batch = config.test_batch_size

    output = torch.empty_like(input)
    for i in range(batch):
        maxvalue = torch.max(input[i, 0, :, :])
        minvalue = torch.min(input[i, 0, :, :])
        delta = maxvalue - minvalue
        output[i, 0, :, :] = (input[i, 0, :, :] - minvalue) / delta

    return output
