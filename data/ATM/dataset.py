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

    def __init__(self, data, include_last_event=True, max_len=500):
        super().__init__()
        self._ids = data['id'].to_numpy()
        self._times = data['time'].to_numpy()
        self._events = (data['event'] + 1).to_numpy()
        self._time_deltas = data['time_delta_scaled'].to_numpy()
        self.include_last_event = include_last_event
        self.max_len = max_len
        self._generate_sequences()

    def _generate_sequences(self):
        timestamps = []
        cat_feats = []
        num_feats = []
        returned = []
        cur_start, cur_end = 0, 1

        # нарезка датасета по последовательностям с одним id и максимальной длиной self.max_len
        while cur_start < len(self._ids):
            if cur_end < len(self._ids) and \
                    self._ids[cur_start] == self._ids[cur_end] and \
                    cur_end - cur_start < self.max_len:
                cur_end += 1
                continue
            else:
                # if cur_end belongs to id of the next ATM:
                if cur_end == len(self._ids) or \
                    self._ids[cur_start] != self._ids[cur_end]:
                    returned.append(torch.BoolTensor([True] * (cur_end - cur_start - 1) + [False]))
                    if self.include_last_event:
                        # append whatever number to align with other cases,
                        # it won't be considered anyway
                        ts = np.concatenate([self._times[cur_start:cur_end],
                                            np.zeros(1)])
                        timestamps.append(torch.FloatTensor(ts))
                        cat_feats.append(torch.LongTensor(self._events[cur_start:cur_end]))
                        num_feats.append(torch.FloatTensor(self._time_deltas[cur_start:cur_end]))
                    else:
                        if cur_start == cur_end - 1:
                            cur_start, cur_end = cur_end, cur_end + 1
                            continue
                        timestamps.append(torch.FloatTensor(self._times[cur_start:cur_end]))
                        cat_feats.append(torch.LongTensor(self._events[cur_start:cur_end - 1]))
                        num_feats.append(torch.FloatTensor(self._time_deltas[cur_start:cur_end - 1]))
                else:
                    returned.append(torch.BoolTensor([True] * (cur_end - cur_start - 1)))
                    timestamps.append(torch.FloatTensor(self._times[cur_start:cur_end]))
                    cat_feats.append(torch.LongTensor(self._events[cur_start:cur_end - 1]))
                    num_feats.append(torch.FloatTensor(self._time_deltas[cur_start:cur_end - 1]))

                cur_start, cur_end = cur_end, cur_end + 1

        self.timestamps = timestamps
        self.cat_feats = list(map(lambda x: x.unsqueeze(-1), cat_feats))
        self.num_feats = list(map(lambda x: x.unsqueeze(-1), num_feats))
        self.returned = returned

    def __getitem__(self, id):
        return self.timestamps[id], \
            self.cat_feats[id], \
            self.num_feats[id], \
            self.returned[id]

    def __len__(self):
        return len(self.timestamps)


def pad_collate_train(batch):
    (timestamps, cat_feats, num_feats, return_mask) = zip(*batch)
    lens = np.array([len(seq) for seq in cat_feats])
    timestamps_padded = pad_sequence(timestamps, batch_first=True, padding_value=0)
    cat_feats_padded = pad_sequence(cat_feats, batch_first=True, padding_value=0)
    num_feats_padded = pad_sequence(num_feats, batch_first=True, padding_value=0)
    return_mask = pad_sequence(return_mask, batch_first=True, padding_value=0)
    padding_mask = (cat_feats_padded == 0).squeeze()

    return timestamps_padded, \
        cat_feats_padded, \
        num_feats_padded, \
        padding_mask, \
        return_mask, \
        lens

class ATMTestDataset(Dataset):
    def __init__(self,
                 data,
                 activity_start,
                 prediction_start,
                 prediction_end,
                 max_len=500):
        super().__init__()
        self.max_len = max_len
        self.prediction_start = prediction_start
        self.prediction_end = prediction_end
        has_event_in_activity = lambda x: ((x['time'] < prediction_start) &
                                           (x['time'] > activity_start)).any()
        valid_ids = data.groupby('id').filter(has_event_in_activity).id.unique()
        self._ids = data['id'][np.isin(data['id'], valid_ids)].to_numpy()
        self._times = data['time'][np.isin(data['id'], valid_ids)].to_numpy()
        self._events = (data['event'] + 1)[np.isin(data['id'], valid_ids)].to_numpy()
        self._time_deltas = data['time_delta_scaled'].to_numpy()
        self._generate_sequences()

    def _generate_sequences(self):
        timestamps = []
        cat_feats = []
        num_feats = []
        targets = []
        cur_start, cur_end = 0, 1

        # нарезка датасета по последовательностям с одним id и максимальной длиной self.max_len
        while cur_start < len(self._ids) - 1:
            if cur_end < len(self._ids) - 1 and \
                    self._ids[cur_start] == self._ids[cur_end] and \
                    self._times[cur_end] <= self.prediction_start:
                cur_end += 1
                continue
            else:
                start_id = max(cur_start, cur_end - self.max_len)
                if self._times[cur_end] <= self.prediction_start or \
                        self._times[cur_end] > self.prediction_end:
                    timestamps.append(torch.FloatTensor(
                        self._times[start_id:cur_end]))
                    cat_feats.append(torch.LongTensor(
                        self._events[start_id:cur_end]))
                    num_feats.append(torch.FloatTensor(
                        self._time_deltas[start_id:cur_end]))
                    targets.append(torch.FloatTensor([-1]))
                else:
                    timestamps.append(torch.FloatTensor(
                        self._times[start_id:cur_end]))
                    cat_feats.append(torch.LongTensor(
                        self._events[start_id:cur_end]))
                    num_feats.append(torch.FloatTensor(
                        self._time_deltas[start_id:cur_end]))
                    targets.append(torch.FloatTensor([self._times[cur_end]]))

                while cur_end < len(self._ids) - 1 and \
                    self._ids[cur_start] == self._ids[cur_end]:
                    cur_start, cur_end = cur_end, cur_end + 1
                cur_start, cur_end = cur_end, cur_end + 1

        self.timestamps = timestamps
        self.cat_feats = list(map(lambda x: x.unsqueeze(-1), cat_feats))
        self.num_feats = list(map(lambda x: x.unsqueeze(-1), num_feats))
        self.targets = targets

    def __getitem__(self, id):
        return self.timestamps[id], \
            self.cat_feats[id], \
            self.num_feats[id], \
            self.targets[id]

    def __len__(self):
        return len(self.timestamps)


def pad_collate_test(batch):
    (timestamps, cat_feats, num_feats, targets) = zip(*batch)
    lens = np.array([len(seq) for seq in timestamps])
    timestamps_padded = pad_sequence(timestamps, batch_first=True, padding_value=0)
    cat_feats_padded = pad_sequence(cat_feats, batch_first=True, padding_value=0)
    num_feats_padded = pad_sequence(num_feats, batch_first=True, padding_value=0)
    targets = torch.FloatTensor(targets).squeeze()

    return timestamps_padded, \
        cat_feats_padded, \
        num_feats_padded, \
        targets, \
        lens


def get_ATM_train_val_loaders(model,
                              activity_start,
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

    assert(model in ['rmtpp', 'rnnsm'], 'Invalid model')
    data = pd.read_csv(filename)
    ids = data.id.unique()
    train_ids, val_ids = train_test_split(ids, train_size=train_ratio, random_state=seed)
    train, val = data[np.isin(data.id, train_ids)], data[np.isin(data.id, val_ids)]
    train_ds = ATMTrainDataset(train, include_last_event=model=='rnnsm', max_len=max_seq_len)
    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=pad_collate_train,
                              drop_last=True)

    val_ds = ATMTestDataset(val, activity_start, prediction_start,
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
