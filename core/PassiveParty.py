"""
Copyright (с) 2024 Guardora.ai
Non-Commercial Open Software License (NCOSL)
"""

import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.Calculator import Calculator
from utils.log import logger
from utils.encryption import load_pub_key, serialize_encrypted_number, load_encrypted_number
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import proto_compiled.service_pb2 as pb2


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class PassiveParty:
    def __init__(self, id: int) -> None:

        Calculator.load_config()

        self.id = id
        self.name = f'party{self.id}'
        self.active_pub_key = None
        self.dataset = None
        self.testset = None
        self.train_status = ''
        self.temp_file = ''
        self.weight = None
        self.num_features = None
        self.scaler = Pipeline([('scaler', StandardScaler())])
        self.train_dataset = None
        self.test_dataset = None
        self.salt = None
        self.dump_model_path = None
        self.path_list = None

    def load_dataset(self, data_path: str, valid_path: str = None, id_col: str = 'ID', frac: float = 0.5):

        dataset = pd.read_csv(data_path)
        dataset = dataset.sample(frac=frac)

        dataset[id_col] = dataset[id_col].astype(str)
        self.dataset = dataset.set_index(id_col)

        if valid_path:
            validset = pd.read_csv(valid_path)
            validset = validset.sample(frac=frac)
            validset[id_col] = validset[id_col].astype(str)
            self.testset = validset.set_index(id_col)

        self.num_features = len([col for col in self.dataset.columns if col != 'y'])

    def recv_active_pub_key(self, pub_key: str):

        self.active_pub_key = load_pub_key(pub_key)
        logger.info(f'{self.name.upper()}: Received public key {pub_key[:10]}. ')
        return pb2.Response(party_name=self.name, status='done')

    def sample_align(self):

        res = pb2.Response(party_name=self.name)

        from utils.sha1 import sha1

        self.train_status = 'Busy'
        train_idx = self.dataset.index.tolist()
        train_idx_map = {sha1(idx): idx for idx in train_idx}
        train_hash = list(train_idx_map.keys())

        map_data = {'train_map': train_idx_map}
        for h in train_hash:
            res.train_hash.append(h)

        if self.testset is not None:
            valid_idx = self.testset.index.tolist()
            valid_idx_map = {sha1(idx): idx for idx in valid_idx}
            valid_hash = list(valid_idx_map.keys())
            map_data['valid_map'] = valid_idx_map
            for h in valid_hash:
                res.valid_hash.append(h)

        map_file = os.path.join(self.path_list[0], f'idx_map.json')
        with open(map_file, 'w+') as f:
            json.dump(map_data, f)

        self.train_status = 'Ready'
        logger.info(f'{self.name.upper()}: Sample hash finished, ready to return. ')

        return res

    def recv_sample_list(self, req):

        self.train_status = 'Busy'
        with open(os.path.join(self.path_list[0], f'idx_map.json'), 'r') as f:
            map_data = json.load(f)

        train_idx = sorted([map_data['train_map'][th] for th in set(req.train_hash)])
        self.dataset = self.dataset.loc[train_idx, :]

        logger.info(f'{self.name.upper()}: Received aligned sample with train length {len(train_idx)}. ')

        if self.testset is not None and req.valid_hash is not None:
            valid_idx = sorted([map_data['valid_map'][vh] for vh in set(req.valid_hash)])
            self.testset = self.testset.loc[valid_idx, :]

        self.train_status = 'Ready'
        return pb2.Response(party_name=self.name, status='done')

    def calc_salt_grad(self, request):

        self.train_status = 'Busy'
        import random
        rng = random.SystemRandom(0)
        self.salt = self.dataset.index.tolist()
        self.salt = [rng.random() for _ in range(self.weight.shape[0] * self.weight.shape[1])]
        self.salt = np.reshape(np.array(self.salt), self.weight.shape)

        grad = pd.Series(np.array(request.grad))
        grad = grad.apply(load_encrypted_number, pub_key=self.active_pub_key)

        grad = np.reshape(grad.values, (self.train_dataset.shape[0], -1))

        logger.info(f'{self.name.upper()}: Gradients received, start calculating on Dataset samples. ')

        res = pb2.Response(party_name=self.name)

        with tqdm(total=self.num_features) as pbar:
            for i in range(self.train_dataset.shape[1]):
                d = pd.Series(self.train_dataset[:, i])
                for j in range(grad.shape[1]):
                    tmp = d * pd.Series(grad[:, j])
                    tmp = tmp.sum()
                    tmp = serialize_encrypted_number(tmp + self.salt[i, j])
                    res.grad.append(res.serialized_encrypted_number(v=tmp['v'], e=tmp['e']))
                pbar.update(1)

        self.train_status = 'Ready'
        logger.info(f'{self.name.upper()}: Calc salt grad is finished, ready to return. ')

        return res

    def calc_updates_for_weight(self, request):

        self.train_status = 'Busy'

        grad = pd.Series(np.array(request.array_float))
        factor = float(request.factor)

        grad = np.reshape(grad, self.weight.shape) - self.salt

        self.weight = self.weight - factor * grad

        self.train_status = 'Ready'
        logger.info(f'{self.name.upper()}: Model weight successfully updated. ')

        return pb2.Response(party_name=self.name, status='done')

    def calc_weighted_data(self, request):

        self.train_status = 'Busy'

        type_of_data = request.type
        n_classes = request.index
        mod = request.model_name

        res = pb2.Response(party_name=self.name)

        if self.weight is None:
            self.dataset['bias'] = [1.0 for i in self.dataset.index]
            self.testset['bias'] = [1.0 for i in self.testset.index]
            self.num_features += 1
            self.weight = 0.1 * np.ones((self.num_features, 1), dtype=np.float32)
            if mod == 'softmax':
                self.weight = 0.1 * np.ones((self.num_features, n_classes), dtype=np.float32)
            self.train_dataset = self.dataset.values
            self.test_dataset = self.testset.values

        if type_of_data == 'train':
            data = self.train_dataset @ self.weight
        else:
            data = self.test_dataset @ self.weight

        data = np.reshape(data, -1)
        for d in data.tolist():
            res.array_float.append(d)

        self.train_status = 'Ready'
        logger.info(f'{self.name.upper()}: Calc weighted dataset and send to active party. ')

        return res

    def set_dump_path(self, path):
        self.dump_model_path = path

    def dump_weight(self, model_name):

        if not os.path.exists(self.dump_model_path):
            os.makedirs(self.dump_model_path)
            logger.info(f'{self.name.upper()}: Model saving dir \'{self.dump_model_path}\' was created. ')

        file_path = os.path.join(self.dump_model_path, model_name)

        data = {'weight': self.weight}
        with open(file_path, 'w+') as f:
            json.dump(data, f, cls=NumpyEncoder)

        logger.info(f'{self.name.upper()}: Model dumped to {file_path}. ')

        return pb2.Response(party_name=self.name, status='done')

    def load_weight(self, model_file_name):

        file_path = os.path.join(self.dump_model_path, model_file_name)
        assert os.path.isfile(file_path), logger.error(f'Error: wrong file_path \'{file_path}\'')

        self.testset['bias'] = [1.0 for i in self.testset.index]

        with open(file_path, 'r') as f:
            model_data = json.load(f)
            self.weight = np.array(model_data['weight'])

        return pb2.Response(party_name=self.name, status='done')

    def empty_res(self):
        return pb2.Response(party_name=self.name, status='empty')
