import torch
import codecs
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)

SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}

def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = torch.iinfo(torch_type).bits // 8
    # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
    # we need to reverse the bytes before we can read them with torch.frombuffer().
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)

def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 1:
        raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
    return x.long()


class MyTrainDataset(Dataset):
    def __init__(self, config, trans):
        self.init_length = config.train_picture_num
        self.ratio = config.img_ratio
        self.img_root = config.train_img_root
        self.label_root = config.train_label_root
        self.init_targets = self._load_label()
        self.targets = self._sift_label()
        self.img_files = self._load_img()
        self.transforms = trans

    def _load_label(self):
        targets = read_label_file(os.path.join(self.label_root + 'train-labels-idx1-ubyte'))
        return targets

    def _load_img(self):
        img_files = sorted(os.listdir(self.img_root), key=lambda i:int(i.split(".")[0]))
        return img_files

    def _sift_label(self):
        label_nums = self.init_length * self.ratio
        self.targets = self.init_targets[0:int(label_nums)]
        return self.targets

    def _norm_tensor(self, array):
        minvalue = torch.min(array)
        maxvalue = torch.max(array)
        delta = maxvalue - minvalue
        array_tensor = (array - minvalue) / delta
        return array_tensor

    def __getitem__(self, index):
        img_path = os.path.join(self.img_root, self.img_files[index])
        img = np.load(img_path)
        img_trans = self._norm_tensor(self.transforms(img))

        targets = int(self.targets[index])

        return img_trans, targets

    def __len__(self):
        return len(self.targets)

class MyTestDataset(Dataset):
    def __init__(self, config, trans):
        self.init_length = config.test_picture_num
        self.img_root = config.test_img_root
        self.label_root = config.test_label_root
        self.init_targets = self._load_label()
        self.targets = self._sift_label()
        self.img_files = self._load_img()
        self.transforms = trans

    def _load_label(self):
        targets = read_label_file(os.path.join(self.label_root + 't10k-labels-idx1-ubyte'))
        return targets

    def _load_img(self):
        img_files = sorted(os.listdir(self.img_root), key=lambda i:int(i.split(".")[0]))
        return img_files

    def _sift_label(self):
        label_nums = self.init_length
        self.targets = self.init_targets[0:int(label_nums)]
        return self.targets

    def _norm_tensor(self, array):
        minvalue = torch.min(array)
        maxvalue = torch.max(array)
        delta = maxvalue - minvalue
        array_tensor = (array - minvalue) / delta
        return array_tensor

    def __getitem__(self, index):
        img_path = os.path.join(self.img_root, self.img_files[index])
        img = np.load(img_path)
        img_trans = self._norm_tensor(self.transforms(img))

        targets = int(self.targets[index])

        return img_trans, targets

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    from utils.Config import get_config_from_json
    import torchvision.transforms as transforms
    config_file = open('../Configuration/Configuration.json')
    config = get_config_from_json(config_file)
    TrainTrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.DMD_pixel_num)
    ])
    my_dataset = MyTrainDataset(config, TrainTrans)

    data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=False)
    for index, (img, label) in enumerate(data_loader):
        print(label)
        img_numpy = img.numpy()[0, 0]
        plt.imshow(img_numpy)
        plt.show()
        print(index)


    print(my_dataset.raw_folder)
    print(0)
