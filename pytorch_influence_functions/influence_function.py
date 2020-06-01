#! /usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd.functional import vhp
from tqdm import tqdm


def s_test(
    x, y, model, samples_loader, gpu=-1, damp=0.01, scale=25.0, recursion_depth=5000
):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        x: torch tensor, test data points, such as test images
        y: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        samples_loader: torch DataLoader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""

    v = grad_z(x, y, model, gpu)
    h_estimate = v

    params, names = make_functional(model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in params)

    # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    for x, y in tqdm(samples_loader):

        if gpu >= 0:
            x, y = x.cuda(), y.cuda()

        def f(*new_params):
            load_weights(model, names, new_params)
            out = model(x)
            loss = calc_loss(out, y)
            return loss

        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]

        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

    load_weights(model, names, params, as_params=True)

    return h_estimate


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    names = []

    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)

    return orig_params, names


def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))


def calc_loss(logits, labels):
    """Calculates the loss

    Arguments:
        logits: torch tensor, input with size (minibatch, nr_of_classes)
        labels: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""

    return F.cross_entropy(logits, labels)


def grad_z(z, t, model, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()

    # initialize
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()

    y = model(z)

    loss = calc_loss(y, t)

    # Compute sum of gradients from model parameters to loss
    return list(grad(loss, list(model.parameters()), create_graph=True))
