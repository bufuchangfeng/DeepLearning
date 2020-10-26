from torch.utils.data import Dataset


class NMTDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return {
            'src': self.src[index]['sentence'],
            'src_len': self.src[index]['len'],
            'trg': self.trg[index]['sentence'],
            'trg_len': self.trg[index]['len']
        }

