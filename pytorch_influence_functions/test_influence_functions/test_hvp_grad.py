from unittest import TestCase
import torch
import pytorch_lightning as pl
from torch.autograd.functional import hessian

from pytorch_influence_functions.influence_functions.hvp_grad import (
    calc_loss,
    s_test_sample,
    grad_z,
    s_test_cg,
)

from pytorch_influence_functions.influence_functions.utils import (
    flatten, load_weights,
    make_functional, split_like,
)

from pytorch_influence_functions.test_influence_functions.utils.logistic_regression import (
    LogisticRegression,
)


class TestIHVPGrad(TestCase):
    def setUp(self) -> None:
        pl.seed_everything(42)

        self.n_features = 10
        self.n_classes = 4

        self.n_params = self.n_classes * self.n_features + self.n_features

        self.model = LogisticRegression(self.n_classes, self.n_features)

        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(gpus=gpus, max_epochs=10)
        trainer.fit(self.model)

        # Setup test point data
        self.test_idx = 8
        self.x_test = torch.tensor([self.model.test_set.data[self.test_idx]], dtype=torch.float)
        self.y_test = torch.tensor([self.model.test_set.targets[self.test_idx]], dtype=torch.long)

        # Compute estimated IVHP
        self.gpu = 1 if torch.cuda.is_available() else -1

        self.train_loader = self.model.train_dataloader(batch_size=10000)
        # Compute anc flatten grad
        grads = grad_z(self.x_test, self.y_test, self.model, gpu=self.gpu)
        flat_grads = flatten(grads)

        # Make model functional
        params, names = make_functional(self.model)
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        flat_params = flatten(params)

        # Initialize Hessian
        h = torch.zeros([flat_params.shape[0], flat_params.shape[0]])

        # Compute real IHVP
        for x_train, y_train in self.train_loader:

            if self.gpu >= 0:
                x_train, y_train = x_train.cuda(), y_train.cuda()

            def f(flat_params_):
                split_params = split_like(params, flat_params_)
                load_weights(self.model, names, split_params)
                out = self.model(x_train)
                loss = calc_loss(out, y_train)
                return loss

            batch_h = hessian(f, flat_params, strict=True)

            with torch.no_grad():
                h += batch_h / float(len(self.train_loader))
        print(h)

        # Make the model back `nn`

        with torch.no_grad():
            load_weights(self.model, names, params, as_params=True)
            inv_h = torch.inverse(h)
            print(inv_h)
            self.real_ihvp = inv_h @ flat_grads

        print(self.real_ihvp)

    def test_s_test_cg(self):
        estimated_ihvp = s_test_cg(
            self.x_test,
            self.y_test,
            self.model,
            self.train_loader,
            damp=0.,
            gpu=self.gpu,
        )

        print("CG")
        self.assertTrue(self.check_estimation(estimated_ihvp))

    def test_calc_s_test_single(self):

        estimated_ihvp = s_test_sample(
            self.model,
            self.x_test,
            self.y_test,
            self.train_loader,
            gpu=self.gpu,
            damp=0.,
            r=10,
            recursion_depth=10_000
        )

        flat_estimated_ihvp = flatten(estimated_ihvp)

        print("LiSSA")
        self.assertTrue(self.check_estimation(flat_estimated_ihvp))

    def check_estimation(self, estimated_ihvp):

        print(estimated_ihvp)
        print(self.real_ihvp / estimated_ihvp)

        with torch.no_grad():
            l_2_difference = torch.norm(self.real_ihvp - estimated_ihvp)
            l_infty_difference = torch.norm(
                self.real_ihvp - estimated_ihvp, p=float("inf")
            )
        print(f"L-2 difference: {l_2_difference}")
        print(f"L-infty difference: {l_infty_difference}")

        return torch.allclose(self.real_ihvp, estimated_ihvp)
