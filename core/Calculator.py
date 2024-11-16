import pandas as pd
import numpy as np
import configparser

from utils.log import logger


class Calculator:

    dict = {}

    lmd = None
    gma = None

    quantile = None

    min_sample = None

    max_depth = None
    max_trees = None

    init_pred = None
    output_thresh = None

    keypair_gen_length = None

    def __init__(self) -> None:
        pass

    @staticmethod
    def grad(preds: pd.Series, trainset: pd.DataFrame) -> (pd.Series, pd.Series):

        assert 'y' in trainset.columns, logger.error('Trainset doesn\'t contain columns \'y\'')
        labels = trainset['y'].values

        preds = 1.0 / (1.0 + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1 - preds)
        logger.debug(f'Gradients: {list(grad.iloc[:5].values)}. ')
        return grad, hess

    @staticmethod
    def split_score_active(grad: pd.Series, hess: pd.Series, left_space: list, right_space: list) -> float:

        left_grad_sum = grad[left_space].sum()
        left_hess_sum = hess[left_space].sum()
        full_grad_sum = grad[left_space + right_space].sum()
        full_hess_sum = hess[left_space + right_space].sum()

        return Calculator.split_score_passive(left_grad_sum, left_hess_sum, full_grad_sum, full_hess_sum)
    
    @staticmethod
    def split_score_passive(left_grad_sum: float, left_hess_sum: float, full_grad_sum: float, full_hess_sum: float) -> float:

        right_grad_sum = full_grad_sum - left_grad_sum
        right_hess_sum = full_hess_sum - left_hess_sum
        
        temp = (left_grad_sum ** 2) / (left_hess_sum + Calculator.lmd)
        temp += (right_grad_sum ** 2) / (right_hess_sum + Calculator.lmd)
        temp -= (full_grad_sum ** 2) / (full_hess_sum + Calculator.lmd)
        return temp / 2 - Calculator.gma

    @staticmethod
    def leaf_weight(grad: pd.Series, hess: pd.Series, instance_space: list) -> float:

        return -grad[instance_space].sum() / (hess[instance_space].sum() + Calculator.lmd)
    
    @staticmethod
    def accuracy(y_true: pd.Series, y_pred: pd.Series, averaging='binary') -> dict:

        if averaging == 'binary':
            y_pred = y_pred.apply(lambda x: 1 if x > Calculator.output_thresh else 0)
        else:
            y_pred = np.argmax(y_pred, axis=1)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=averaging),
            'recall': recall_score(y_true, y_pred, average=averaging),
            'f1': f1_score(y_true, y_pred, average=averaging)
        }
    
    @staticmethod
    def load_config(config_path=r"config/config.conf"):
        cfg = configparser.ConfigParser()
        cfg.read(config_path)

        Calculator.dict = cfg['passive_parties_list']['dict']

        Calculator.lin_lr = float(cfg['params.train']['lin_lr'])
        Calculator.log_lr = float(cfg['params.train']['log_lr'])
        Calculator.soft_lr = float(cfg['params.train']['soft_lr'])
        Calculator.epochs = int(cfg['params.train']['epochs'])

        Calculator.output_thresh = float(cfg['params.predict']['output_thresh'])
        Calculator.keypair_gen_length = int(cfg['encryption']['key_bitlength'])
        logger.info(f'Config loaded. ')

    @staticmethod
    def generate_config(config_path='config/config.conf'):
        cfg = configparser.ConfigParser()
        cfg['passive_parties_list'] = {
            'dict': {'1': {'ip': '127.0.0.1', 'port': 50051}}
        }
        cfg['params.train'] = {
            'lin_lr': 0.05,
            'log_lr': 0.15,
            'soft_lr': 0.1,
            'epochs': 10
        }
        cfg['params.predict'] = {
            'output_thresh': 0.5
        }
        cfg['encryption'] = {
            'key_bitlength': 512
        }
        with open(config_path, 'w+') as config_file:
            cfg.write(config_file)


if __name__ == '__main__':
    Calculator.generate_config()
