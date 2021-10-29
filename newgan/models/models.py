import torch
from os.path import abspath


class Config(object):
    def __init__(self, lr=1e-3, d_steps=1, g_steps=1, batch_size=64, z_dim=1,
                 g_dim=128, d_dim=128, beta1=0.5, beta2=0.999,
                 lambda_gp=10, lambda_rec=1, lambda_kl=0.5,
                 num_workers=0, epochs=100,
                 device=None):
        self.lr = lr
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_gp = lambda_gp
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl
        self.num_workers = num_workers
        self.epochs = epochs
        self.project_root = abspath(__file__).split('/newgan')[0]
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
