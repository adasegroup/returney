import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os

class ATMDataset(Dataset):
    def __init__(self, path, max_len=500):
        super().__init__()
        self.max_len = max_len
        data = pd.read_csv(path)
        self.ids = data['id']
        self.times = data['time']
        self.events = data['event'] + 1
        self.time_seqs, self.event_seqs = self.generate_sequence()

    def generate_sequence(self):
        time_seqs = []
        event_seqs = []
        cur_start, cur_end = 0, 1
        
        # нарезка датасета по последовательностям с одним id и максимальной длиной self.max_len
        while cur_start < len(self.ids):
            if cur_end < len(self.ids) and \
                    self.ids[cur_start] == self.ids[cur_end] and \
                    cur_end - cur_start < self.max_len:
                cur_end += 1
                continue
            else:
                time_seqs.append(torch.Tensor(self.times[cur_start:cur_end].to_numpy()))
                event_seqs.append(torch.LongTensor(self.events[cur_start:cur_end].to_numpy()))
                cur_start, cur_end = cur_end, cur_end + 1

        return time_seqs, event_seqs

    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    
def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  return xx_pad[..., None], yy_pad[..., None], x_lens, y_lens

def get_ATM_loader(path=None, batch_size=32, test=False, max_seq_len=500):
    if path is None:
        filename = '../data/ATM/train_day.csv' if not test else '../data/ATM/test_day.csv'
    else:
        filename = path
    ds = ATMDataset(filename, max_len=max_seq_len)
    loader = DataLoader(dataset=ds, 
                        batch_size=batch_size,
                        shuffle=True if not test else False,
                        collate_fn=pad_collate,
                        drop_last=True)
    return loader