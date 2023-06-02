import torch

path = '../Benchmark/net_state_dict_best_3layers_0_ps76miu_60region_complex_mask_imgLoss_NoAd_layer2&3.ckpt'
new_path = '../Benchmark/net_state_dict_best_3layers_0_ps76miu_60region_complex_mask_imgLoss_NoAd_layer3.ckpt'
model_dict = torch.load(path, map_location=torch.device('cpu'))

new_state_dict = {}

for key, value in model_dict.items():
    # print(key + ':' + str(value.numpy()))

    if 'real2' not in key and 'imag2' not in key and 'alpha1' not in key and 'beta1' not in key:
        new_state_dict[key] = value
    # if 'real1' not in key and 'imag1' not in key:
    #     new_state_dict[key] = value
    # if 'real0' not in key and 'imag0' not in key:
    #     new_state_dict[key] = value

for keys, values in new_state_dict.items():
    print(keys + ':' + str(values.numpy()))

torch.save(new_state_dict, new_path)
print(0)