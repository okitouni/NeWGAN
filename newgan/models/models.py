import torch
from os.path import abspath


class Config(object):
    def __init__(
        self,
        lr=1e-3,
        d_steps=1,
        g_steps=1,
        batch_size=64,
        z_dim=1,
        g_dim=128,
        d_dim=128,
        beta1=0.5,
        beta2=0.999,
        lambda_gp=10,
        lambda_rec=1,
        lambda_kl=0.5,
        num_workers=0,
        epochs=100,
        device=None,
    ):
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
        self.project_root = abspath(__file__).split("/newgan")[0]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gradient_penalty(discriminator, real_data, fake_data):
    eps = torch.rand(real_data.shape[0], 1)
    eps = eps.expand(real_data.size())
    interpolated = eps * real_data + (1 - eps) * fake_data
    interpolated.requires_grad_(True)
    pred = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
