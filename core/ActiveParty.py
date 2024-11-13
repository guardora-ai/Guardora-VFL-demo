import os
import json
from typing import Any
import time

import numpy as np
import pandas as pd
from phe import paillier, PaillierPublicKey

from core.Calculator import Calculator
from utils.log import logger
from utils.encryption import serialize_pub_key, serialize_encrypted_number, load_encrypted_number
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import grpc
import proto_compiled.service_pb2 as pb2
import proto_compiled.service_pb2_grpc as pb2_grpc


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


class ActiveParty:
    def __init__(self, mod, config_path) -> None:
        Calculator.load_config(config_path)

        self.mod = mod

        self.passive_parties = eval(Calculator.dict)

        self.id = 0
        self.name = f'party{self.id}'
        self.pub_key, self.pri_key = paillier.generate_paillier_keypair(n_length=Calculator.keypair_gen_length)
        logger.info(f'{self.name.upper()}: Paillier key generated. ')

        self.dataset = None
        self.testset = None

        self.model = None

        if self.mod == 'logistic':
            self.model = Model_LogReg(lr=Calculator.log_lr)
        elif self.mod == 'softmax':
            self.model = Model_SoftReg(lr=Calculator.soft_lr)
        else:
            self.model = Model_LinReg(lr=Calculator.lin_lr)

        self.model_path = None

        self.passive_port = {}
        self.reverse_passive_port = {}
        self.cur_preds = None
        self.diff_pred_target = None

        self.weight = None
        self.num_features = None
        self.epochs = Calculator.epochs
        self.train_loss = None
        self.n_classes = 2
        self.scaler = Pipeline([('scaler', StandardScaler())])

    def load_dataset(self, data_path: str, test_path: str = None, id_col: str = 'ID', target: str = 'y', frac: float = 0.5):

        dataset = pd.read_csv(data_path)

        dataset = dataset.sample(frac=frac)

        dataset[id_col] = dataset[id_col].astype(str)
        dataset = dataset.rename(columns={target: 'y'})
        self.dataset = dataset.set_index(id_col)

        if test_path:
            testset = pd.read_csv(test_path)
            testset = testset.sample(frac=frac)
            testset[id_col] = testset[id_col].astype(str)
            testset = testset.rename(columns={target: 'y'})
            self.testset = testset.set_index(id_col)

        self.cur_preds = pd.Series(Calculator.init_pred, index=self.dataset.index)
        self.num_features = len([col for col in self.dataset.columns if col != 'y'])

    def broadcast_pub_key(self):

        pub_key = serialize_pub_key(self.pub_key)

        def send_pub_key():
            for _, value in self.passive_parties.items():
                ip = value['ip']
                port = value['port']
                with grpc.insecure_channel(f"{ip}:{port}") as channel:
                    stub = pb2_grpc.MVP_VFLStub(channel)
                    response = stub.request_analysis(pb2.Request(party_name=self.name, processor='recvActivePubKey', pub_key=pub_key))
                    self.passive_port[response.party_name] = [ip, port]
                    self.reverse_passive_port[str(ip) + ':' + str(port)] = response.party_name

        send_pub_key()

        logger.info(f'{self.name.upper()}: Public key sending to all passive parties. ')

    def sample_align(self):

        from utils.sha1 import sha1

        train_idx = self.dataset.index.tolist()
        train_idx_map = {sha1(idx): idx for idx in train_idx}
        train_hash = set(train_idx_map.keys())

        if self.testset is not None:
            valid_idx = self.testset.index.tolist()
            valid_idx_map = {sha1(idx): idx for idx in valid_idx}
            valid_hash = set(valid_idx_map.keys())

        def init_sample_align():
            nonlocal train_hash, valid_hash
            logger.debug(f'{self.name.upper()}: Initiating sample alignment. ')
            for _, value in self.passive_parties.items():
                ip = value['ip']
                port = value['port']
                with grpc.insecure_channel(f"{ip}:{port}") as channel:
                    stub = pb2_grpc.MVP_VFLStub(channel)
                    response = stub.request_analysis(pb2.Request(party_name=self.name, processor='initSampleAlign'))
                    train_hash = train_hash.intersection(set(response.train_hash))
                    if self.testset is not None and response.valid_hash is not None:
                        valid_hash = valid_hash.intersection(set(response.valid_hash))

                    logger.info(f'{self.name.upper()}: Received sample list from {response.party_name}')

        init_sample_align()

        logger.info(
            f'{self.name.upper()}: Sample alignment finished. Intersect trainset contains {len(train_hash)} samples. ')

        req = pb2.Request(party_name=self.name, processor='recvSampleList')
        for h in train_hash:
            req.train_hash.append(h)

        if self.testset is not None:
            for h in valid_hash:
                req.valid_hash.append(h)

        def send_aligned_sample(request):
            logger.debug(f'{self.name.upper()}: Sending aligned sample. ')
            for _, value in self.passive_parties.items():
                ip = value['ip']
                port = value['port']
                with grpc.insecure_channel(f"{ip}:{port}") as channel:
                    stub = pb2_grpc.MVP_VFLStub(channel)
                    stub.request_analysis(request)

        send_aligned_sample(req)

        logger.info(f'{self.name.upper()}: Aligned sample sending to all passive parties. ')

        self.dataset = self.dataset.loc[[train_idx_map[th] for th in train_hash], :]
        if self.testset is not None:
            self.testset = self.testset.loc[[valid_idx_map[vh] for vh in valid_hash], :]

    def decrypt_salt_grad(self, salt_grad):

        s_grad = pd.Series(np.array(salt_grad))
        s_grad = s_grad.apply(load_encrypted_number, pub_key=self.pub_key)

        def decrypt_data(data):
            dec_data = self.pri_key.decrypt(data)
            return dec_data

        dec_salt_grad = s_grad.apply(decrypt_data)
        return dec_salt_grad

    def train(self):
        self.broadcast_pub_key()
        self.sample_align()

        if self.mod == 'linear' or self.mod == 'logistic' or self.mod == 'softmax':

            train_target = self.dataset['y'].values
            test_target = self.testset['y'].values
            self.dataset['bias'] = [1.0 for i in self.dataset.index]
            self.testset['bias'] = [1.0 for i in self.testset.index]
            self.num_features += 1

            if self.mod == 'softmax':
                self.n_classes = len(np.unique(train_target))
                self.num_features = (self.num_features, self.n_classes)
                train_target = pd.get_dummies(train_target).to_numpy().astype(float)
            else:
                self.num_features = (self.num_features, 1)
                train_target = np.expand_dims(train_target, -1)
                test_target = np.expand_dims(test_target, -1)

            self.model.init_weight(self.num_features)

            train_dataset = self.dataset.drop(['y'], axis=1).values
            train_dataset = self.scaler.fit_transform(train_dataset)

            test_dataset = self.testset.drop(['y'], axis=1).values
            test_dataset = self.scaler.fit_transform(test_dataset)

            for epoch in range(self.epochs):

                self.cur_preds = self.model.weighted_data(train_dataset)

                def init_weighted_data(type_proc):
                    logger.debug(f'{self.name.upper()}: Initiating calc weighted data. ')
                    for _, value in self.passive_parties.items():
                        ip = value['ip']
                        port = value['port']
                        with grpc.insecure_channel(f"{ip}:{port}") as channel:
                            stub = pb2_grpc.MVP_VFLStub(channel)
                            response = stub.request_analysis(pb2.Request(party_name=self.name, index=self.n_classes,
                                                                         processor='initWeightedData', type=type_proc,
                                                                         model_name=self.mod))
                            w_data = np.array(response.array_float)
                            w_data = np.reshape(w_data, self.cur_preds.shape)
                            self.cur_preds = self.cur_preds + w_data

                init_weighted_data('train')

                self.cur_preds = self.model.get_pred(self.cur_preds)
                self.diff_pred_target = self.model.get_cur_diff(self.cur_preds, train_target)

                grad = self.model.get_grad(train_dataset.shape[0], np.transpose(train_dataset), self.diff_pred_target)
                self.train_loss = self.model.get_loss(self.cur_preds, train_target)                

                self.model.update_weight(grad)

                self.model.update_gradients(self.diff_pred_target, self.pub_key)

                req = pb2.Request(party_name=self.name, processor='recvGradientsReg', index=self.n_classes)

                for elem in self.model.grad_enc:
                    req.grad.append(req.serialized_encrypted_number(v=elem['v'], e=elem['e']))

                def send_gradients(request):
                    logger.info(f'Sending gradients. ')
                    for _, value in self.passive_parties.items():
                        ip = value['ip']
                        port = value['port']
                        with grpc.insecure_channel(f"{ip}:{port}") as channel:
                            stub = pb2_grpc.MVP_VFLStub(channel)
                            response = stub.request_analysis(request)
                            logger.info(f'{self.name.upper()}: Received salt grad from {response.party_name}. ')

                            dec_salt_grad = self.decrypt_salt_grad(response.grad)

                            req = pb2.Request(party_name=self.name, processor='recvDecryptedGradients')
                            for d in dec_salt_grad:
                                req.array_float.append(d)
                            req.lr = self.model.lr
                            req.factor = self.model.factor
                            response = stub.request_analysis(req)

                send_gradients(req)

                logger.info(f'{self.name.upper()}: Gradients broadcasted to all passive parties. ')

                try:
                    self.cur_preds = self.model.weighted_data(test_dataset)

                    init_weighted_data('test')

                    test_preds = self.model.get_pred(self.cur_preds)

                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    if self.mod == 'softmax':
                        test_loss = self.model.get_loss(test_preds,
                                                        pd.get_dummies(test_target).to_numpy().astype(float))
                        test_preds = np.argmax(test_preds, axis=1)
                        averaging = 'weighted'
                    else:
                        test_loss = self.model.get_loss(test_preds, test_target)
                        test_preds = pd.Series(test_preds[:, 0]).apply(lambda x: 1 if x > Calculator.output_thresh else 0)
                        averaging = 'binary'

                    logger.info(f'{self.name.upper()}: Epoch {epoch}, Train_loss = {self.train_loss:.3f}, '
                                f'Test_loss = {test_loss:.3f}, Test accuracy: {accuracy_score(test_target, test_preds):.3f}, '
                                f'Test precision: {precision_score(test_target, test_preds, average=averaging):.3f}, '
                                f'Test recall: {recall_score(test_target, test_preds, average=averaging):.3f}, '
                                f'Test f1: {f1_score(test_target, test_preds, average=averaging):.3f}')
                except:
                    pass

            def init_dump_model():
                self.model_path = time.strftime(f'{self.mod}_model%y%m%d.json', time.localtime())
                logger.debug(f'{self.name.upper()}: Initiating dump trained model parameters for all passive parties. ')

                for _, value in self.passive_parties.items():
                    ip = value['ip']
                    port = value['port']
                    with grpc.insecure_channel(f"{ip}:{port}") as channel:
                        stub = pb2_grpc.MVP_VFLStub(channel)
                        stub.request_analysis(pb2.Request(party_name=self.name, processor='recvDumpModel',
                                                          model_name=self.model_path))

            init_dump_model()
            logger.info(f'Training completed. ')
        else:
            logger.error(f'{self.name.upper()}: Wrong mod \'{self.mod}\'. Please, choose one from \'linear\', '
                         f'\'logistic\', \'softmax\'. ')

    def dump_model(self, file_path: str):

        dir_path, file_name = file_path, self.model_path

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.error(f'{self.name.upper()}: Model saving path \'{dir_path}\' was created. ')

        if not file_name:
            self.model_path = time.strftime(f'{self.mod}_model%y%m%d.json', time.localtime())
            file_path = os.path.join(dir_path, self.model_path)
        else:
            file_path = os.path.join(dir_path, file_name)

        with open(file_path, 'w+') as f:
            json.dump(self.model.dump(), f, cls=NumpyEncoder)
        logger.info(f'{self.name.upper()}: Model dumped to {file_path}. ')

        return file_path

    def load_model(self, file_path: str):

        assert os.path.isfile(file_path), logger.error(f'Error: wrong file_path \'{file_path}\'')

        with open(file_path, 'r') as f:
            model_data = json.load(f)
        if self.mod == 'linear' or self.mod == 'logistic' or self.mod == 'softmax':
            self.model.load(model_data)
            self.n_classes = self.model.weight.shape[1]

        _, self.model_path = os.path.split(file_path)

        logger.info(f'{self.name.upper()}: Model loaded from {file_path}. ')

    def predict(self):

        if self.testset is None:
            logger.error(f'{self.name.upper()}: No testset loaded. ')
            return

        logger.info(f'{self.name.upper()}: Start predicting. ')
        averaging = 'binary'

        if self.mod == 'linear' or self.mod == 'logistic' or self.mod == 'softmax':

            def init_predict():
                logger.debug(f'{self.name.upper()}: Initiating loading model parameters by all passive parties for predicting. ')
                for _, value in self.passive_parties.items():
                    ip = value['ip']
                    port = value['port']
                    with grpc.insecure_channel(f"{ip}:{port}") as channel:
                        stub = pb2_grpc.MVP_VFLStub(channel)
                        response = stub.request_analysis(pb2.Request(party_name=self.name, processor='recvLoadModel',
                                                                     model_name=self.model_path))
            init_predict()

            self.testset['bias'] = [1.0 for i in self.testset.index]
            test_dataset = self.testset.drop(['y'], axis=1).values
            test_dataset = self.scaler.fit_transform(test_dataset)

            self.cur_preds = self.model.weighted_data(test_dataset)

            def init_weighted_data(type_proc):
                logger.debug(f'{self.name.upper()}: Initiating calc weighted data. ')
                for _, value in self.passive_parties.items():
                    ip = value['ip']
                    port = value['port']
                    with grpc.insecure_channel(f"{ip}:{port}") as channel:
                        stub = pb2_grpc.MVP_VFLStub(channel)
                        response = stub.request_analysis(pb2.Request(party_name=self.name, processor='initWeightedData',
                                                                     type=type_proc, index=self.n_classes))

                        logger.info(f'{self.name.upper()}: Received train data from {response.party_name}. ')
                        w_data = np.array(response.array_float)
                        w_data = np.reshape(w_data, self.cur_preds.shape)
                        self.cur_preds = self.cur_preds + w_data

            init_weighted_data('test')

            preds = self.model.get_pred(self.cur_preds)
            if self.mod == 'softmax':
                averaging = 'weighted'
            else:
                preds = pd.Series(preds[:, 0])

            acc = Calculator.accuracy(self.testset["y"], preds, averaging)
            logger.info(f'{self.name.upper()}: Test accuracy: '
                        f'accuracy = {acc["accuracy"]:.3f}, precision = {acc["precision"]:.3f}, '
                        f'recall = {acc["recall"]:.3f}, f1 = {acc["f1"]:.3f}')

            logger.info(f'{self.name.upper()}: All finished. ')
        else:
            logger.error(f'{self.name.upper()}: Wrong mod \'{self.mod}\'. Please, choose one from \'linear\', '
                         f'\'logistic\', \'softmax\'. ')


class Model_LinReg:
    def __init__(self, active_idx=0, lr=0.01) -> None:
        self.eps = 1e-15
        self.lr = lr
        self.factor = 2.
        self.grad = None
        self.grad_enc = None
        self.weight = None

    def __len__(self):
        return self.weight.shape()[0]

    def __getitem__(self, idx):
        return self.weight[idx]

    def init_weight(self, dim):
        self.weight = 0.1 * np.ones(dim, dtype=np.float32)

    def update_weight(self, g):
        self.weight -= (self.lr * g)

    def weighted_data(self, data):
        return np.matmul(data, self.weight)

    def get_pred(self, pred):
        return pred

    def get_cur_diff(self, pred, target):
        return pred - target

    def get_grad(self, n, data, diff):
        return (self.factor / n) * (data @ diff)

    def get_loss(self, pred, target):
        diff = pred - target
        return np.mean(diff * diff)

    def update_gradients(self, g, pub_key):
        self.grad = pd.Series(g[:, 0])
        self.encrypt_gradients(pub_key)

    def encrypt_gradients(self, pub_key: PaillierPublicKey):
        from tqdm import tqdm

        logger.info(f'Gradients encrypting... ')

        with tqdm(total=len(self.grad)) as pbar:
            def encrypt_data(data, pub_key: PaillierPublicKey):
                pbar.update(1)
                enc_data = pub_key.encrypt(data)
                return serialize_encrypted_number(enc_data)

            self.grad_enc = self.grad.apply(encrypt_data, pub_key=pub_key)

    def dump(self) -> dict[str, Any]:
        data = {'weight': self.weight}
        return data

    def load(self, data: dict):
        self.weight = np.array(data['weight'])


class Model_LogReg:
    def __init__(self, active_idx=0, lr=0.01) -> None:
        self.eps = 1e-15
        self.lr = lr
        self.factor = 1.
        self.grad = None
        self.grad_enc = None
        self.weight = None

    def __len__(self):
        return self.weight.shape()[0]

    def __getitem__(self, idx):
        return self.weight[idx]

    def init_weight(self, dim):
        self.weight = 0.1 * np.ones(dim, dtype=np.float32)

    def update_weight(self, g):
        self.weight -= (self.lr * g)

    def weighted_data(self, data):
        return np.matmul(data, self.weight)

    def get_pred(self, pred):
        return np.power(np.exp(-1. * pred) + 1., -1)

    def get_cur_diff(self, pred, target):
        return pred - target

    def get_grad(self, n, data, diff):
        return (self.factor / n) * (data @ diff)

    def get_loss(self, pred, target):
        return np.mean(-1 * (target * np.log(pred + self.eps) + (1. - target) * np.log(1 - pred + self.eps)))

    def update_gradients(self, g, pub_key):
        self.grad = pd.Series(g[:, 0])
        self.encrypt_gradients(pub_key)

    def encrypt_gradients(self, pub_key: PaillierPublicKey):
        from tqdm import tqdm

        logger.info(f'Gradients encrypting... ')

        with tqdm(total=len(self.grad)) as pbar:
            def encrypt_data(data, pub_key: PaillierPublicKey):
                pbar.update(1)
                enc_data = pub_key.encrypt(data)
                return serialize_encrypted_number(enc_data)

            self.grad_enc = self.grad.apply(encrypt_data, pub_key=pub_key)

    def dump(self) -> dict[str, Any]:
        data = {'weight': self.weight}
        return data

    def load(self, data: dict):
        self.weight = np.array(data['weight'])


class Model_SoftReg:
    def __init__(self, active_idx=0, lr=0.01) -> None:
        self.grad = None
        self.grad_enc = None
        self.weight = None
        self.factor = 1
        self.lr = lr
        self.eps = 1e-15

    def __len__(self):
        return self.weight.shape()[0]

    def __getitem__(self, idx):
        return self.weight[idx]

    def init_weight(self, dim):
        self.weight = 0.1 * np.ones(dim, dtype=np.float32)

    def weighted_data(self, data):
        return np.matmul(data, self.weight)

    def update_weight(self, g):
        self.weight -= (self.lr * g)

    def get_pred(self, pred):
        exp = np.exp(pred)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def get_cur_diff(self, pred, target):
        return pred - target

    def get_grad(self, n, data, diff):
        return (self.factor / n) * (data @ diff)

    def get_loss(self, pred, target):
        loss_train = -1 * np.mean(np.sum(target * np.log(pred + self.eps), axis=1))
        return loss_train

    def update_gradients(self, g, pub_key):
        self.grad = pd.Series(np.reshape(g, [-1]))
        self.encrypt_gradients(pub_key)

    def encrypt_gradients(self, pub_key: PaillierPublicKey):
        from tqdm import tqdm

        logger.info(f'Gradients encrypting... ')

        with tqdm(total=len(self.grad)) as pbar:
            def encrypt_data(data, pub_key: PaillierPublicKey):
                pbar.update(1)
                enc_data = pub_key.encrypt(data)
                return serialize_encrypted_number(enc_data)

            self.grad_enc = self.grad.apply(encrypt_data, pub_key=pub_key)

    def dump(self) -> dict[str, Any]:
        data = {'weight': self.weight}
        return data

    def load(self, data: dict):
        self.weight = np.array(data['weight'])
