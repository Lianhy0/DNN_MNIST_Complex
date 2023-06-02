import numpy as np



def AccuracyFunction(config, Model, TestLoader, state='accu'):
    """
    AccuracyFunction aims to calculate the accuracy of current training model.
    :param config: Configuration
    :param Model: Training Model
    :param TestLoader: TestLoader
    :return: return the accuracy
    """
    correct = 0
    for index, (data, target) in enumerate(TestLoader):
        print(index)
        # if Train:
        #     data[data >= 0.5] = 1
        #     data[data < 0.5] = 0
        data = data.to(config.device)
        target = target.to(config.device)
        output = Model(config, data, state='accu', checkparameter='phase')
        outputnumpy = output.cpu().detach().numpy()
        classpredict = np.argmax(outputnumpy)
        targetnumpy = target.cpu().detach().numpy()

        if targetnumpy == classpredict:
            correct = correct + 1

    Acc = correct / config.test_picture_num
    print(Acc)
    return Acc
