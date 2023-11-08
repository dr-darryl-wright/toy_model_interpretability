import torch
import itertools
import numpy as np
import torch.nn as nn
import torch_optimizer as optim
# import matplotlib.pyplot as plt

from copy import deepcopy
# from timer import timeit
from sacred import Experiment
from model_helper import model_builder


ReLU = torch.nn.ReLU()

print('GPU:', torch.cuda.get_device_name(0))

ex = Experiment('toy_model_interpretability')


def single_feature_activations(d, data, setup):  # Computes the activations for each feature on its own
    m = d['0.weight'].shape[1]
    k = d['0.weight'].shape[0]
    N = setup['N']
    nonlinearity = data['nonlinearity']

    vectors = torch.eye(N, device='cuda')
    embedder = setup['fixed_embedder']
    inputs = torch.matmul(vectors,embedder.T)
    model = model_builder(d['2.weight'].shape[0], m, k, nonlinearity)
    model.to('cuda')
    model.load_state_dict(d)
    outputs = model[:2].forward(inputs).T
    return outputs


@torch.jit.script
def sample_vectors_power_law(N: int, eps: float, batch_size: int, embedder):
    """
        Generates random uniform vectors in a tensor of shape (N,batch_size)
        with sparsity 1-eps. These are returned as v.

        Applies embedding matrix to v to produce a low-dimensional embedding,
        returned as x.
    """
    v = torch.rand((int(batch_size), int(N)), device='cuda')

    compare = 1. / torch.arange(1, int(N)+1, device='cuda')**1.1
    compare *= N * eps / torch.sum(compare)

    compare[compare >= 1] = 1

    sparsity = torch.bernoulli(compare.repeat(int(batch_size), 1))
                
    v *= sparsity
    x = torch.matmul(v, embedder.T)  # Embeds features in a low-dimensional space

    return v, x


@torch.jit.script
def sample_vectors_equal(N: int, eps: float, batch_size: int, embedder):
    """
        Generates random uniform vectors in a tensor of shape (N,batch_size)
        with sparsity 1-eps. These are returned as v.

        Applies embedding matrix to v to produce a low-dimensional embedding,
        returned as x.
    """

    v = torch.rand((int(batch_size), int(N)), device='cuda')
    
    compare = eps * torch.ones((int(batch_size), int(N)), device='cuda')
    sparsity = torch.bernoulli(compare)
            
    v *= sparsity
    x = torch.matmul(v,embedder.T) # Embeds features in a low-dimensional space

    return v, x


def gen_inputs_no_feat_vec(N, feats, n_feats):
    v = torch.rand((n_feats, N, N), device='cuda')
    # print(v.shape)

    compare = torch.linspace(0, 1, N, device='cuda') * torch.ones((N, N), device='cuda')
    compare = compare.T.unsqueeze(0).repeat(n_feats, 1, 1)
    sparsity = torch.bernoulli(compare)
    # print(sparsity.shape)

    v *= sparsity
    v[range(n_feats), :, feats] = 0

    return torch.Tensor(v > 0).type(torch.float32)
#     return v

@torch.jit.script
def loss_func(batch_size: int, outputs, vectors):
    loss = torch.sum((outputs - vectors)**2) / batch_size
    return loss


@torch.jit.script
def abs_loss_func(batch_size: int, outputs, vectors):
    loss = torch.sum((outputs - torch.abs(vectors))**2) / batch_size
    return loss


def train(setup, model, training_steps, _run):
    penalisation_type = setup['penalisation_type']
    m = setup['m']
    k = setup['k']
    N = setup['N']
    eps = setup['eps']
    learning_rate = setup['learning_rate']
    batch_size = setup['batch_size']
    fixed_embedder = setup['fixed_embedder']
    task = setup['task']
    decay = setup['decay']
    reg = setup['reg']

    if task == 'autoencoder':
        l_func = loss_func
        sample_vectors = setup['sampler']
    elif task == 'random_proj':
        l_func = loss_func
        def sample_vectors(N, eps, batch_size, fixed_embedder):
            v, i = setup['sampler'](N, eps, batch_size, fixed_embedder)
            v = torch.matmul(v, setup['output_embedder'].T)
            return v, i
    elif task == 'abs':
        l_func = abs_loss_func
        # I need to cut eps in half to make this equivalent density.
        # Different samples have different sparse choices so doubles the density.
        def sample_vectors(N, eps, batch_size, fixed_embedder):
            v1, i1 = setup['sampler'](N, eps / 2, batch_size, fixed_embedder)
            v2, i2 = setup['sampler'](N, eps / 2, batch_size, fixed_embedder)
            return v1 - v2, i1 - i2
    else:
        print('Task not recognized. Exiting.')
        exit()

    c0, c1 = [], []
    for c in list(itertools.combinations(range(N), 2)):
        c0.append(c[0])
        c1.append(c[1])

    c0 = torch.tensor(c0).to('cuda')
    c1 = torch.tensor(c1).to('cuda')
    nc = len(c0)

    optimizer = optim.Lamb(model.parameters(), lr=setup['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**9, eta_min=0)

    losses = []
    models = []
    total_losses = []
    penalised_term_losses = []
    # first_term_losses = []
    # second_term_losses = []

    # Training loop
    for i in range(training_steps):
        optimizer.zero_grad(set_to_none=True)

        vectors, inputs = sample_vectors(N, eps, batch_size, fixed_embedder)
        outputs = model.forward(inputs)
        activations = model[:2].forward(inputs)

        vs = torch.eye(N, device='cuda')
        ins = torch.matmul(vs, fixed_embedder.T)
        sfa = model[:2].forward(ins).T

        if penalisation_type in ['all_for_one', 'frac_act', 'sfa_abofa_dot']:
            all_but_one_vs = torch.ones(N, N, device='cuda') - torch.eye(N, device='cuda')
            ins = torch.matmul(all_but_one_vs, fixed_embedder.T)
            abofa = model[:2].forward(ins).T

        if penalisation_type in ['frac_act']:
            # calculate the sparse random feature activations
            v = torch.rand((int(N), int(N)), device='cuda')
            compare = torch.linspace(0, 1, N, device='cuda') * torch.ones((int(N), int(N)), device='cuda')
            sparsity = torch.bernoulli(compare).T
            v *= sparsity
            sparse_random_vs = torch.Tensor(v > 0).type(torch.float32)
            ins = torch.matmul(sparse_random_vs, fixed_embedder.T)
            srfa = model[:2].forward(ins).T

        # get the maximum activation for each feature and divide by the sum of activations for that feature
        frac_mono = torch.amax(sfa, dim=[0]) / (1e-10 + torch.sum(sfa, dim=[0]))
        # divide the maximum activations per feature by the maximum activation for any feature
        frac_mono = frac_mono / torch.amax(frac_mono)
        ### My implementation of monosemanticity measure ###
        if penalisation_type == 'frac_mono':
            # r = 1 - torch.sum(frac_mono) / batch_size
            r = - torch.sum(frac_mono) / N
        ### My attempt at a penalisation term ###
        elif penalisation_type == 'hard_dot':
            sfa = torch.gt(sfa, 0) * 1.  # * 1. to convert from bool to float implements a hard measure
            # print(sfa.shape)
            # dot = torch.inner(torch.t(sfa), torch.t(sfa)) / (2. * N)
            dot = torch.inner(sfa, sfa) # k-by-k the diagonal counts how many features a hidden neuron activates for.
            # print(dot)
            # print(dot.shape)
            # r = torch.sum(dot) - torch.sum(torch.diagonal(dot))
            # calculate mask to only include neruons firing for more than 1 feature.
            mask = torch.ne(torch.diagonal(dot), 1.) * 1.
            # print(mask)
            # print(torch.sum(mask))
            # masking rather than simply subtracting k to avoid encouraging the network to create all "dead" neurons
            # sum up the diagonal elements that fire for more than one neuron
            # print(torch.sum(torch.diagonal(dot)), torch.sum(torch.diagonal(dot) * mask))
            r = torch.sum(torch.diagonal(dot) * mask) / (N * k)
            # print(torch.sum(torch.diagonal(dot)) - torch.sum(torch.diagonal(dot) * mask))
            # r = torch.sum(torch.diagonal(dot)) / (N * k)
            # r = r / ((N*N - N) / 2.)
            if i % 2 ** 4 == 0:
                _run.log_scalar('training.number_PNs', torch.sum(mask), i)
                _run.log_scalar('training.mean_features_PN_firing', torch.sum(torch.diagonal(dot) * mask), i)
        elif penalisation_type == 'soft_dot':
            # raise NotImplementedError
            # dot = torch.inner(torch.t(sfa), torch.t(sfa)) #/ (2. * N)
            # r = torch.sum(dot) - torch.sum(torch.diagonal(dot))
            # r = r / ((N*N - N) / 2.)
            # r = r / torch.sum(r)  # normalise
            # r /= 2.
            # dot = torch.inner(sfa, sfa)
            dot = torch.inner(sfa, sfa)
            dot = torch.sum(torch.tril(dot, -1))
            r = dot / N * (N - 1)
        elif penalisation_type == 'soft_dot_choose_2':
            # sfa = sfa / (N * (1e-10 + torch.sum(sfa, axis=-1).repeat(N, 1).T))
            # sfa = sfa / (1e-10 + torch.sum(sfa, axis=-1).repeat(N, 1).T)
            # a = sfa.index_select(0, c0)
            # b = sfa.index_select(0, c1)
            # r = (a - b) * (a - b)
            # r = torch.sum(r, axis=-1)
            # r = torch.sqrt(r)
            # r = torch.sum(r) / len(c0)
            sfa = sfa / (1e-10 + torch.max(sfa, dim=0)[0].repeat(k, 1))
            a = sfa.index_select(1, c0)
            b = sfa.index_select(1, c1)
            # r = (a - b) ** 2
            r = a * b
            # r = torch.sum(r, dim=[0])
            # r = torch.sqrt(r)
            r = torch.sum(r) / nc
        elif penalisation_type == 'soft_dot_choose_2_by_batch':
            """
                This doesn't work since for each training example all features are > 0. I cannot think of a good way
                to decide which pairs should be included in the penalisation or even how they should be penalised. But
                randomly subsampling pairs and penalising based on the dot product doesn't make sense.
            """
            raise NotImplementedError
            # order = torch.randperm(batch_size)
            # print(activations.shape)
            # print(activations[order].shape)
            # print(activations[order].T.shape)
            # exit()
            # acts = activations[order].T / (1e-10 + torch.max(activations[order].T, dim=0)[0].repeat(k, 1))
            a = acts.index_select(1, c0)
            b = acts.index_select(1, c1)
            r = a * b
            r = torch.sum(r) / nc
        elif penalisation_type == 'soft_dot_choose_2_rand_act':
            """
                This is hopefully an improvement on soft_dot_choose_2 that will penalise polysemanticity across the
                dynamic range of inputs from 0 to 1 and not just when the input is 1. This is the effect I was trying 
.            """
            rvs = torch.eye(N, device='cuda') * torch.rand(N, device='cuda')
            rins = torch.matmul(rvs, fixed_embedder.T)
            rsfa = model[:2].forward(rins).T
            rsfa = rsfa / (1e-10 + torch.max(rsfa, dim=0)[0].repeat(k, 1))
            a = rsfa.index_select(1, c0)
            b = rsfa.index_select(1, c1)
            r = a * b
            r = torch.sum(r) / nc
        elif penalisation_type == 'all_for_one':
            unit_sfa = torch.Tensor(sfa > 0).type(torch.float32)
            unit_abofa = torch.Tensor(abofa > 0).type(torch.float32)
            unit_sfa_mono_mask = torch.Tensor(torch.sum(unit_sfa, dim=1) == 1).type(torch.float32).repeat(N, 1).T
            unit_abofa_mono_mask = torch.Tensor(torch.sum(unit_abofa, dim=1) == 1).type(torch.float32).repeat(N, 1).T
            r = 1 - (torch.sum((unit_sfa * unit_sfa_mono_mask) * (unit_abofa * unit_abofa_mono_mask))) / k
        elif penalisation_type == 'frac_act':
            unit_sfa = torch.Tensor(sfa > 0).type(torch.float32)
            unit_abofa = torch.Tensor(abofa > 0).type(torch.float32)
            unit_srfa = torch.Tensor(srfa > 0).type(torch.float32)
            r = ((torch.sum(unit_sfa) + torch.sum(unit_abofa) + torch.sum(unit_srfa)) - 3*k) / (k*N)
        elif penalisation_type == 'jacob':
            n_test_feats = N
            test_feats = torch.randint(0, N, size=(n_test_feats, ))
            single_max_val, single_max_ind = torch.max(sfa[test_feats], dim=1)
            feat_rmvd_rand_inps = gen_inputs_no_feat_vec(N, test_feats, n_test_feats)
            feat_rmvd_embed = torch.matmul(feat_rmvd_rand_inps, fixed_embedder.T)
            feat_rmvd_hidden = model[:2].forward(feat_rmvd_embed)
            corr_neuron_acts = feat_rmvd_hidden[range(n_test_feats), :, single_max_ind]
            r = torch.sum(corr_neuron_acts) / (n_test_feats * N)
        elif penalisation_type == 'sfa_abofa_dot':
            # print(sfa.shape)
            # print(torch.max(sfa, dim=1)[0].repeat((N, 1)).shape)
            normed_sfa = torch.nan_to_num(sfa / torch.max(sfa, dim=1)[0].repeat((N, 1)).T)
            normed_abofa = torch.nan_to_num(abofa / torch.max(abofa, dim=1)[0].repeat((N, 1)).T)
            first_term = torch.sum(torch.tril(torch.matmul(normed_sfa, normed_sfa.T), -1)) / (N * (N - 1))
            second_term = torch.sum(torch.diag(torch.matmul(normed_sfa, normed_abofa.T))) / N
            r = first_term + second_term
        elif penalisation_type == 'baseline':
            r = 0.
        elif penalisation_type == 'l1_baseline':
            r = 0.
        else:
            raise NotImplementedError

        l = l_func(batch_size, outputs, vectors)
        # print(r)
        # l = l + r
        loss = l + r + reg * torch.sum(torch.abs(activations)) / batch_size
        # loss = r
        # print(i, loss)

        loss.backward()

        optimizer.step()
        scheduler.step()

        if i < training_steps / 2:
            state = model.state_dict()
            state['0.bias'] *= (1 - decay * learning_rate)
            model.load_state_dict(state)

        if i % 2 ** 4 == 0:  # Avoids wasting time on copying the scalar over
            _run.log_scalar('training.mean_frac_mono', float(torch.sum(frac_mono) / N), i)
            losses.append(float(l))
            _run.log_scalar('training.target_loss', float(l), i)
            total_losses.append(float(loss))
            _run.log_scalar('training.total_loss', float(loss), i)
            penalised_term_losses.append(float(r))
            _run.log_scalar('training.penalised_loss', float(r), i)
            # print(float(r))
            if penalisation_type == 'sfa_abofa_dot':
                _run.log_scalar('training.first_term_loss', float(first_term), i)
                # penalised_term_losses.append(float(r))
                _run.log_scalar('training.second_term_loss', float(second_term), i)

        if (i & (i + 1) == 0) and (i + 1) != 0:  # Checks if i is a power of 2
            models.append(deepcopy(model))

    os = {
        'k': setup['k'],
        'log2_batch_size': setup['log2_batch_size'],
        'log2_training_steps': setup['log2_training_steps'],
        'learning_rate': setup['learning_rate'],
        'sample_kind': setup['sample_kind'],
        'initial_bias': setup['init_bias'],
        'nonlinearity': setup['nonlinearity'],
        'losses': losses,
        'final_model': model.state_dict(),
        'log2_spaced_models': list(m.state_dict() for m in models),
        'setup': setup,
        'task': task,
        'decay': decay,
        'eps': eps,
        'm': m,
        'N': N,
        'reg': reg
    }

    return losses, total_losses, penalised_term_losses, model, models, os



def make_random_embedder(N,m):
    matrix = np.random.randn(N,m) #  Make a random matrix that's (N,m)
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    # Now u is a matrix (N,m) with orthogonal columns and nearly-orthogonal rows
    # Normalize the rows of u
    u /= (np.sum(u**2,axis=1)**0.5)[:, np.newaxis]
    t = torch.tensor(u.T, requires_grad=False, device='cuda', dtype=torch.float)
    return t


class SoLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.exp(input)


@ex.config
def config():
    penalisation_type = None
    m = None
    k = None
    N = None
    task = None
    feature_dist = None
    sample_kind = None
    eps = None
    learning_rate = None
    decay = 0.
    init_bias = 0.
    reg = 0.
    log2_batch_size = None
    log2_training_steps = None
    nonlinearity = 'ReLU'


# @timeit
@ex.automain
def run(penalisation_type, m, k, N, task, feature_dist, sample_kind, eps, learning_rate, decay,
        init_bias, reg, log2_batch_size, log2_training_steps, nonlinearity, _run):

    assert (feature_dist in ['uniform', 'power-law'])

    assert penalisation_type in ['baseline', 'l1_baseline', 'hard_dot', 'soft_dot', 'soft_dot_choose_2',
                                 'soft_dot_choose_2_by_batch', 'soft_dot_choose_2_rand_act', 'frac_mono',
                                 'all_for_one', 'frac_act', 'jacob', 'sfa_abofa_dot']

    if sample_kind == 'equal':
        sampler = sample_vectors_equal
    elif sample_kind == 'power_law':
        sampler = sample_vectors_power_law
    else:
        print('Sample kind not recognized. Exiting.')
        exit()

    setup = {
        'penalisation_type': penalisation_type,
        'N': N,
        'm': m,
        'k': k,
        'log2_batch_size': log2_batch_size,
        'log2_training_steps': log2_training_steps,
        'batch_size': 2**log2_batch_size,
        'learning_rate': learning_rate,
        'eps': eps,
        'fixed_embedder': make_random_embedder(N, m),
        'init_bias': init_bias,
        'nonlinearity': nonlinearity,
        'sample_kind': sample_kind,
        'sampler': sampler,
        'task': task,
        'decay': decay,
        'reg': reg
    }

    if task == 'random_proj':
        setup['output_embedder'] = make_random_embedder(N, m)
        
    if nonlinearity == 'ReLU':
        activation = nn.ReLU()
    elif nonlinearity == 'GeLU':
        activation = nn.GELU()
    elif nonlinearity == 'SoLU':
        activation = SoLU()
    else:
        print('No valid activation specified. Quitting.')
        exit()

    if task == 'random_proj':
        output_dim = m
    else:
        output_dim = N

    model = torch.jit.script(
                torch.nn.Sequential(
                    nn.Linear(m, k, bias=True),
                    activation,
                    nn.Linear(k, output_dim, bias=False)
                )
        ).to('cuda')

    state = model.state_dict()
    state['0.bias'] += init_bias
    model.load_state_dict(state)
                
    losses, total_losses, penalised_term_losses, model, models, outputs = train(setup, model, 2**log2_training_steps, _run)

    del setup['sampler']

    if 'baaseline' in penalisation_type:
        fname = f"./my_models/{penalisation_type}_model_{task}_{nonlinearity}_k_{k}_batch_{log2_batch_size}_steps_{log2_training_steps}_learning_rate_{learning_rate}_sample_{sample_kind}_init_bias_{init_bias}_decay_{decay}_eps_{eps}_m_{m}_N_{N}_reg_{reg}.pt"
    else:
        fname = f"./my_models/{penalisation_type}_penalised_model_{task}_{nonlinearity}_k_{k}_batch_{log2_batch_size}_steps_{log2_training_steps}_learning_rate_{learning_rate}_sample_{sample_kind}_init_bias_{init_bias}_decay_{decay}_eps_{eps}_m_{m}_N_{N}_reg_{reg}.pt"
    _run.add_artifact(fname)
    torch.save(outputs, fname)

    # plt.figure(figsize=(10, 10))
    # plt.plot(np.log2(np.arange(1, 2**4*len(losses)+1, 2**4)), penalised_term_losses,
    #          label=f'{penalisation_type} penalised term')
    # plt.plot(np.log2(np.arange(1, 2**4*len(losses)+1,2 **4)), losses, label='target loss')
    # plt.plot(np.log2(np.arange(1, 2**4*len(losses)+1, 2**4)), total_losses, label='total loss')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()

    return losses, total_losses, penalised_term_losses, setup  #, model, models


# if __name__ == '__main__':
#     ex.run_commandline()

#
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("penalisation_type", type=str)
# parser.add_argument("k", type=int)
# parser.add_argument("log2_batch_size", type=int)
# parser.add_argument("log2_training_steps", type=int)
# parser.add_argument("learning_rate", type=float)
# parser.add_argument("sample_kind")
# parser.add_argument("init_bias", type=float)
# parser.add_argument("nonlinearity")
# parser.add_argument("task")
# parser.add_argument("decay", type=float)
# parser.add_argument("eps", type=float)
# parser.add_argument("m", type=int)
# parser.add_argument("N", type=int)
# parser.add_argument("reg", type=float)
#
# args = parser.parse_args()
#
# penalisation_type = args.penalisation_type
# k = args.k
# learning_rate = args.learning_rate
# log2_batch_size = args.log2_batch_size
# log2_training_steps = args.log2_training_steps
# sample_kind = args.sample_kind
# init_bias = args.init_bias
# nonlinearity = args.nonlinearity
# task = args.task
# decay = args.decay
# eps = args.eps
# m = args.m
# N = args.N
# reg = args.reg

# assert penalisation_type in ['dot', 'soft_dot', 'frac_mono']
#
# data = run(penalisation_type,
#            N,
#            m,
#            args.k,
#            eps,
#            2**log2_batch_size,
#            learning_rate,
#            2**log2_training_steps,
#            sample_kind,
#            init_bias,
#            nonlinearity,
#            task,
#            decay,
#            reg
#            )

# losses, total_losses, penalised_term_losses, model, models, setup = data

# del setup['sampler']

# fname = f"./my_models/{penalisation_type}_penalised_model_{task}_{nonlinearity}_k_{k}_batch_{log2_batch_size}_steps_{log2_training_steps}_learning_rate_{learning_rate}_sample_{sample_kind}_init_bias_{init_bias}_decay_{decay}_eps_{eps}_m_{m}_N_{N}_reg_{reg}.pt"
# outputs = {
#     'penalisation_type': penalisation_type,
#     'k': k,
#     'log2_batch_size': log2_batch_size,
#     'log2_training_steps': log2_training_steps,
#     'learning_rate': learning_rate,
#     'sample_kind': sample_kind,
#     'initial_bias': init_bias,
#     'nonlinearity': nonlinearity,
#     'losses': losses,
#     'final_model': model.state_dict(),
#     'log2_spaced_models': list(m.state_dict() for m in models),
#     'setup': setup,
#     'task': task,
#     'decay': decay,
#     'eps': eps,
#     'm': m,
#     'N': N,
#     'reg': reg
# }
#
#
# torch.save(outputs, fname)
#
# plt.figure(figsize=(10,10))
# plt.plot(np.log2(np.arange(1,2**4*len(losses)+1,2**4)), penalised_term_losses,
#          label=f'{penalisation_type} penalised term')
# plt.plot(np.log2(np.arange(1,2**4*len(losses)+1,2**4)), losses, label='target loss')
# plt.plot(np.log2(np.arange(1,2**4*len(losses)+1,2**4)), total_losses, label='total loss')
# plt.yscale('log')
# plt.legend()
# plt.show()
