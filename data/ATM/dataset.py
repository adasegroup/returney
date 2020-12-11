import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch
import os


class ATMTrainDataset(Dataset):
    def get_indexes(path):
        data = pd.read_csv(path)
        return data['id'].values
  
    def __init__(self, path, ids=None, max_len=500):
        super().__init__()
        self.max_len = max_len
        data = pd.read_csv(path)
        self.ids = data['id'][np.isin(data['id'], ids)].to_numpy()
        self.times = data['time'][np.isin(data['id'], ids)].to_numpy()
        self.events = (data['event'] + 1)[np.isin(data['id'], ids)].to_numpy()
        self.time_seqs, self.time_deltas_seqs, self.event_seqs = self._generate_sequences()

    def _generate_sequences(self):
        time_seqs = []
        event_seqs = []
        time_deltas_seqs = []
        cur_start, cur_end = 0, 1

        # нарезка датасета по последовательностям с одним id и максимальной длиной self.max_len
        while cur_start < len(self.ids):
            if cur_end < len(self.ids) and \
                    self.ids[cur_start] == self.ids[cur_end] and \
                    cur_end - cur_start < self.max_len:
                cur_end += 1
                continue
            else:
                time_seqs.append(torch.Tensor(self.times[cur_start:cur_end]))
                event_seqs.append(torch.LongTensor(self.events[cur_start:cur_end]))
                cur_start, cur_end = cur_end, cur_end + 1
                
                time_deltas_seqs.append(torch.cat([
                    torch.zeros(1),
                     time_seqs[-1][1:] - time_seqs[-1][:-1]],
                     dim=0))

        return time_seqs, time_deltas_seqs, event_seqs

    def __getitem__(self, id):
        return self.time_seqs[id], self.time_deltas_seqs[id], self.event_seqs[id]

    def __len__(self):
        return len(self.time_seqs)


class ATMTestDataset(Dataset):
    def __init__(self,
                 path,
                 ids,
                 activity_start,
                 prediction_start,
                 prediction_end,
                 max_len=500):
        super().__init__()
        self.max_len = max_len
        self.prediction_start = prediction_start
        self.prediction_end = prediction_end
        data = pd.read_csv(path)
        has_event_in_activity = lambda x: ((x['time'] < prediction_start) &
            (x['time'] > activity_start)).any()
        valid_ids = data.groupby('id').filter(has_event_in_activity).id.unique()
        valid_ids = np.intersect1d(valid_ids, ids)
        self.ids = data['id'][np.isin(data['id'], valid_ids)].to_numpy()
        self.times = data['time'][np.isin(data['id'], valid_ids)].to_numpy()
        self.events = (data['event'] + 1)[np.isin(data['id'], valid_ids)].to_numpy()
        self.time_seqs, self.time_deltas_seqs, self.targets, self.event_seqs = self._generate_sequences()
        

    def _generate_sequences(self):
        time_seqs = []
        targets = []
        event_seqs = []
        time_deltas_seqs = []
        cur_start, cur_end = 0, 1

        # нарезка датасета по последовательностям с одним id и максимальной длиной self.max_len
        while cur_start < len(self.ids):
            if cur_end < len(self.ids) and \
                    self.ids[cur_start] == self.ids[cur_end] and \
                    self.times[cur_end] <= self.prediction_start:
                cur_end += 1
                continue
            else:
                if self.times[cur_end - 1] <= self.prediction_start or \
                    self.times[cur_end - 1] > self.prediction_end:
                    time_seqs.append(torch.Tensor(
                        self.times[cur_end-self.max_len:cur_end]))
                    targets.append(torch.Tensor([-1]))
                    event_seqs.append(torch.LongTensor(
                        self.events[cur_start-self.max_len:cur_end]))
                    
                    time_deltas_seqs.append(torch.cat([
                        torch.zeros(1),
                         time_seqs[-1][1:] - time_seqs[-1][:-1]],
                         dim=0))
                else:
                    time_seqs.append(torch.Tensor(
                        self.times[cur_end-self.max_len-1:cur_end-1]))
                    targets.append(torch.Tensor([self.times[cur_end-1]]))
                    event_seqs.append(torch.LongTensor(
                        self.events[cur_start-self.max_len-1:cur_end-1]))
                    
                    time_deltas_seqs.append(torch.cat([
                        torch.zeros(1),
                         time_seqs[-1][1:] - time_seqs[-1][:-1]],
                         dim=0))
                cur_start, cur_end = cur_end, cur_end + 1

        return time_seqs, time_deltas_seqs, targets, event_seqs

    def __getitem__(self, id):
        return self.time_seqs[id], self.time_deltas_seqs[id], self.targets[id], self.event_seqs[id]

    def __len__(self):
        return len(self.time_seqs)


def pad_collate_train(batch):
  (time_seqs, time_deltas_seqs, event_seqs) = zip(*batch)
  lens = [len(seq) for seq in time_seqs]
  time_seqs_padded = pad_sequence(time_seqs, batch_first=True, padding_value=0)
  event_seqs_padded = pad_sequence(event_seqs, batch_first=True, padding_value=0)
  time_deltas_seqs_padded = pad_sequence(time_deltas_seqs, batch_first=True, padding_value=0)

  return time_seqs_padded[..., None], time_deltas_seqs_padded[..., None], event_seqs_padded[..., None], lens


def pad_collate_test(batch):
  (time_seqs, time_deltas_seqs, targets, event_seqs) = zip(*batch)
  lens = [len(seq) for seq in time_seqs]
  time_seqs_padded = pad_sequence(time_seqs, batch_first=True, padding_value=0)
  event_seqs_padded = pad_sequence(event_seqs, batch_first=True, padding_value=0)
  time_deltas_seqs_padded = pad_sequence(time_deltas_seqs, batch_first=True, padding_value=0)
  targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
  return time_seqs_padded[..., None], time_deltas_seqs_padded[..., None], targets_padded, event_seqs_padded[..., None], lens


def get_ATM_train_val_loaders(activity_start,
                              prediction_start,
                              prediction_end,
                              path=None,
                              batch_size=32,
                              max_seq_len=500,
                              train_ratio=0.7,
                              seed=42):
    if path is None:
        filename = 'data/ATM/train_day.csv'
    else:
        filename = path

    csv = pd.read_csv(filename)
    ids = csv.id.unique()
    train_ids, val_ids = train_test_split(ids, train_size=train_ratio, random_state=seed)
    train_ds = ATMTrainDataset(filename, train_ids, max_len=max_seq_len)
    train_loader = DataLoader(dataset=train_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=pad_collate_train,
                        drop_last=True)

    val_ds = ATMTestDataset(filename, val_ids, activity_start, prediction_start,
                            prediction_end)
    val_loader = DataLoader(dataset=val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=pad_collate_test,
                        drop_last=False)

    return train_loader, val_loader


def get_ATM_test_loader(activity_start,
                        prediction_start,
                                    prediction_end,
                        path=None, 
                        batch_size=32, 
                        max_seq_len=500):
    if path is None:
        filename = 'test_day.csv'
    else:
        filename = path
    
    csv = pd.read_csv(filename)
    ids = csv.id.unique()
    
    test_ds = ATMTestDataset(filename, ids, activity_start, prediction_start,
                            prediction_end)
    test_loader = DataLoader(dataset=test_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=pad_collate_test,
                        drop_last=False)

    return test_loader
