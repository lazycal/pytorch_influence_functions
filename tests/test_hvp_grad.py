from unittest import TestCase
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from torch.autograd.functional import hessian
from sklearn.linear_model import LogisticRegression as SklearnLogReg
import unittest
import numpy as np
from tqdm import tqdm
from pytorch_influence_functions.influence_functions.influence_functions import calc_influence_single

from pytorch_influence_functions.influence_functions.hvp_grad import (
    s_test_sample,
    grad_z,
    s_test_cg,
    s_test_sample_exact,
    
)

from pytorch_influence_functions.influence_functions.utils import (
    load_weights,
    make_functional,
    tensor_to_tuple,
    parameters_to_vector,
)
from utils.dummy_dataset import (
    DummyDataset,
)

from utils.logistic_regression import (
    LogisticRegression,
)
from utils.vis import (
    visualize_result,
)
def make_loss_f(model, params, names, x, y, wd=0):
    def f(flat_params_):
        split_params = tensor_to_tuple(flat_params_, params)
        load_weights(model, names, split_params)
        out = model(x)
        loss = model.loss(out, y) #calc_loss(out, y) + wd/2 * torch.sum(flat_params_*flat_params_)
        return loss
    return f

class TestIHVPGrad(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pl.seed_everything(0)

        cls.n_features = 10
        cls.n_classes = 3

        cls.n_params = cls.n_classes * cls.n_features + cls.n_features

        cls.wd = wd = 1e-2 # weight decay=1/(nC)
        cls.model = LogisticRegression(cls.n_classes, cls.n_features, wd=cls.wd)

        gpus = 1 if torch.cuda.is_available() else 0
        
        trainer = pl.Trainer(gpus=gpus, max_epochs=10)
        # trainer.fit(self.model)

        use_sklearn = True
        if use_sklearn:
            cls.train_dataset = cls.model.training_set #DummyDataset(cls.n_features, cls.n_classes)
            multi_class = "multinomial" if cls.model.n_classes != 2 else "auto"
            clf = SklearnLogReg(C=1/len(cls.train_dataset)/wd, tol=1e-8, max_iter=1000, multi_class=multi_class)

            clf.fit(cls.train_dataset.data, cls.train_dataset.targets)

            with torch.no_grad():
                cls.model.linear.weight = torch.nn.Parameter(
                    torch.tensor(clf.coef_, dtype=torch.float)
                )
                cls.model.linear.bias = torch.nn.Parameter(
                    torch.tensor(clf.intercept_, dtype=torch.float)
                )

        # Setup test point data
        cls.test_idx = 5
        cls.x_test = torch.tensor(
            cls.model.test_set.data[[cls.test_idx]], dtype=torch.float
        )
        cls.y_test = torch.tensor(
            cls.model.test_set.targets[[cls.test_idx]], dtype=torch.long
        )

        # Compute estimated IVHP
        cls.gpu = 1 if torch.cuda.is_available() else -1

        if cls.gpu >= 0:
            cls.model = cls.model.cuda()
            cls.x_test = cls.x_test.cuda()
            cls.y_test = cls.y_test.cuda()

        cls.train_loader = cls.model.train_dataloader(batch_size=40000)
        # Compute anc flatten grad
        grads = grad_z(cls.x_test, cls.y_test, cls.model, gpu=cls.gpu)
        flat_grads = parameters_to_vector(grads)

        print("Grads:")
        print(flat_grads)

        # Make model functional
        params, names = make_functional(cls.model)
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        flat_params = parameters_to_vector(params)

        # Initialize Hessian
        h = torch.zeros([flat_params.shape[0], flat_params.shape[0]])
        if cls.gpu == 1:
            h = h.cuda()

        # Compute real IHVP
        for x_train, y_train in cls.train_loader:

            if cls.gpu >= 0:
                x_train, y_train = x_train.cuda(), y_train.cuda()

            f = make_loss_f(cls.model, params, names, x_train, y_train, wd=wd)

            batch_h = hessian(f, flat_params, strict=True)

            with torch.no_grad():
                h += batch_h / float(len(cls.train_loader))

        h = (h + h.transpose(0,1))/2
        print("Hessian:")
        print(h)

        np.save("hessian_pytorch.npy", h.cpu().numpy())
        from numpy import linalg as LA
        ei = LA.eig(h.cpu().numpy())[0]
        print('ei=', ei)
        print("max,min eigen value=", ei.max(), ei.min())
        assert ei.min() > 0, "Error: Non-positive Eigenvalues"

        # Make the model back `nn`

        with torch.no_grad():
            load_weights(cls.model, names, params, as_params=True)
            inv_h = torch.inverse(h)
            print("Inverse Hessian")
            print(inv_h)
            cls.real_ihvp = inv_h @ flat_grads

        print("Real IHVP")
        print(cls.real_ihvp)

    def test_s_test_cg(self):
        estimated_ihvp = s_test_cg(
            self.x_test,
            self.y_test,
            self.model,
            self.train_loader,
            damp=0.0,
            gpu=self.gpu,
        )

        print("CG")
        self.assertTrue(self.check_estimation(estimated_ihvp))

    def test_s_test_sample(self):

        estimated_ihvp = s_test_sample(
            self.model,
            self.x_test,
            self.y_test,
            self.train_loader,
            gpu=self.gpu,
            damp=0.0,
            r=10,
            recursion_depth=10_000,
            batch_size=500,
        )

        flat_estimated_ihvp = parameters_to_vector(estimated_ihvp)

        print("LiSSA")
        self.assertTrue(self.check_estimation(flat_estimated_ihvp))

        print("Influence")
        inf_app, inf_rea = [], []
        for i, (x_train, y_train) in enumerate(self.model.train_dataloader(batch_size=1, shuffle=False)):
            grads_train = grad_z(x_train, y_train, self.model, gpu=self.gpu)
            flat_grads_train = parameters_to_vector(grads_train)
            inf_app.append(- torch.sum(flat_grads_train * flat_estimated_ihvp / len(self.model.training_set)).item())
            inf_rea.append(- torch.sum(flat_grads_train * self.real_ihvp / len(self.model.training_set)).item())
        np.save("influence.npy", {'inf_app':inf_app, 'inf_rea':inf_rea})

    def test_calc_influence_single(self):
        test_loader = DataLoader(self.model.test_set)
        loss_diff_approx, harmful, _, _ = calc_influence_single(self.model, self.train_loader, test_loader, self.test_idx, 
            gpu=self.gpu,
            recursion_depth=10_000,
            r=10,
            damp=0.0,
            batch_size=500,
            # exact=True
        )

        sample_num = 100
        ## calculating real changes
        torch_model = LogisticRegression(self.n_classes, self.n_features, wd=self.wd)
        if self.gpu >= 0: torch_model = torch_model.cuda()
        loss_diff_true = []
        test_loss_ori = self.model.loss(self.model(self.x_test), self.y_test).item()
        sample_indice = harmful[:sample_num]+harmful[-sample_num:]
        for i, index in enumerate(tqdm(sample_indice)):
            x_train, y_train = self.train_dataset.data, self.train_dataset.targets
            # get minus one dataset
            x_train_minus_one = np.delete(x_train, index, axis=0)
            y_train_minus_one = np.delete(y_train, index, axis=0)

            # retrain
            C = 1.0 / ((len(x_train) - 1) * self.wd)
            multi_class = "multinomial" if self.model.n_classes != 2 else "auto"
            sklearn_model_minus_one = SklearnLogReg(C=C, tol=1e-8, max_iter=1000, multi_class=multi_class)
            sklearn_model_minus_one.fit(x_train_minus_one, y_train_minus_one.ravel())
            print('LBFGS training took {} iter.'.format(sklearn_model_minus_one.n_iter_))

            # assign w on tensorflow model
            with torch.no_grad():
                torch_model.linear.weight = torch.nn.Parameter(
                    torch.tensor(sklearn_model_minus_one.coef_, dtype=torch.float)
                )
                torch_model.linear.bias = torch.nn.Parameter(
                    torch.tensor(sklearn_model_minus_one.intercept_, dtype=torch.float)
                )
            
            if self.gpu >= 0:
                torch_model = torch_model.cuda()

            # get retrain loss
            test_loss_retrain = torch_model.loss(torch_model(self.x_test), self.y_test).item()

            # get true loss diff
            loss_diff_true.append(test_loss_retrain - test_loss_ori)

            print('Original loss       :{}'.format(test_loss_ori))
            print('Retrain loss        :{}'.format(test_loss_retrain))
            print('True loss diff      :{}'.format(loss_diff_true[i]))
            print('Estimated loss diff :{}'.format(loss_diff_approx[index]))
        r2_score = visualize_result(loss_diff_true, loss_diff_approx[sample_indice])
        print('r2_score=',r2_score)

        self.assertTrue(r2_score > 0.9)

    def test_s_test_sample_exact(self):

        estimated_ihvp = s_test_sample_exact(
            self.model,
            self.x_test,
            self.y_test,
            self.train_loader,
            gpu=self.gpu,
        )

        flat_estimated_ihvp = parameters_to_vector(estimated_ihvp)

        print("Exact")
        self.assertTrue(self.check_estimation(flat_estimated_ihvp))

    def check_estimation(self, estimated_ihvp):

        print(estimated_ihvp)
        print("real / estimate")
        print(self.real_ihvp / estimated_ihvp)

        with torch.no_grad():
            l_2_difference = torch.norm(self.real_ihvp - estimated_ihvp)
            l_infty_difference = torch.norm(
                self.real_ihvp - estimated_ihvp, p=float("inf")
            )
        print(f"L-2 difference: {l_2_difference}")
        print(f"L-infty difference: {l_infty_difference}")

        return torch.allclose(self.real_ihvp, estimated_ihvp)


if __name__ == "__main__":
    unittest.main()
