import os
import csv
import matplotlib

import matplotlib.cm as cmx
import matplotlib.pyplot as plt

from copy import deepcopy
from model_helper import *
from matplotlib import colors
from collections import Counter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter


font = {
        'weight' : 'normal',
        'size'   : 18
        }
matplotlib.rc('font', **font)

nice_names = {
    'initial_bias': 'Initial Mean Bias',
    'learning_rate': 'Learning Rate',
    'k': '# of Neurons',
    'decay': 'Weight Decay Rate',
    'eps': r'$\epsilon$',
    'reg': 'L1 Regularization Strength'
}


def training_plot(batch, sweep_var, log_color=True, cm='Blues', loss_range=None):   
    sweep_vars = list(b[sweep_var] for b in batch)
    if log_color:
        sweep_vars_col = np.log10(sweep_vars)
    else:
        sweep_vars_col = sweep_vars

    sweep_ran_col = max(sweep_vars_col) - min(sweep_vars_col)
    
    ncol = 1 + int(len(batch) / 4)
    
    vmin = min(sweep_vars_col) - 0.3*sweep_ran_col
    vmax = max(sweep_vars_col) + 0.1*sweep_ran_col
    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0, width_ratios=[0.96,0.04], wspace=0)
    gs2 = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0, width_ratios=[0.97,0.03], wspace=0)

    fig = plt.figure(figsize=(13,9))
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[2,0])
    
    cax = fig.add_subplot(gs2[:,1])

    plot_data = []
    for j in range(len(batch)):
        plot_data.append([])
        for i in range(len(batch[j]['log2_spaced_models'])):
            sfa = single_feature_activations(batch[j]['log2_spaced_models'][i], batch[j], batch[j]['setup'])
            frac_mono = np.amax(sfa,axis=1) / (1e-10 + np.sum(sfa,axis=1))
            mean_bias = torch.mean(batch[j]['log2_spaced_models'][i]['0.bias']).numpy()
            plot_data[-1].append((i,sum(frac_mono > r_threshold), mean_bias))

    plot_data = np.array(plot_data)
    for j in range(len(batch)):
        col = scalarMap.to_rgba(sweep_vars_col[j])
        ax1.plot(np.log2(np.arange(1,2**4*len(batch[j]['losses'])+1,2**4)), batch[j]['losses'], label=f'{sweep_vars[j]}', c=col)
        ax2.plot(plot_data[j][:,0],plot_data[j][:,1]/512, c=col)
        ax3.plot(plot_data[j][:,0],plot_data[j][:,2], c=col)

    if loss_range is not None:
        ax1.set_ylim(loss_range)
        
    ax1.set_xticks([])
    ax2.set_xticks([])

    ax1.set_ylabel('Loss')
    ax2.set_ylabel('# Mono Neurons\n / # Features')
    ax3.set_ylabel('Mean bias')

    if log_color:
        clabel = r'$\log_{10}$'+f'{nice_names[sweep_var]}'
    else:
        clabel = str(nice_names[sweep_var])

    cb = fig.colorbar(scalarMap, cax=cax, orientation='vertical', label=clabel)


    ax2.axhline(1, linestyle=':', c='k')
    ax3.set_xlabel(r'$\log_2 \mathrm{Training\ Steps}$')
    
    ax1.set_title(f'Sweeping {nice_names[sweep_var]}')
    return fig

def plot_bias(batch, sweep_var, log_color=True, cm='Blues'):
    sweep_vars = list(b[sweep_var] for b in batch)
    if log_color:
        sweep_vars_col = np.log10(sweep_vars)
    else:
        sweep_vars_col = sweep_vars

    sweep_ran_col = max(sweep_vars_col) - min(sweep_vars_col)
    
    ncol = 1 + int(len(batch) / 4)
    
    cNorm  = colors.Normalize(vmin=min(sweep_vars_col) - 0.3*sweep_ran_col, vmax=max(sweep_vars_col) + 0.1*sweep_ran_col)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    fig = plt.figure(figsize=(13,9))

    plt.axhline(0, c='k', linestyle=':')

    for i in range(len(batch)):
        sfa = single_feature_activations(batch[i]['log2_spaced_models'][-1], batch[i], batch[i]['setup'])

        # Sort the neurons to put the most-monosemantic first
        inds = np.argsort(-np.amax(sfa,axis=1) / (1e-10 + np.mean(sfa,axis=1)))

        bias = batch[i]['log2_spaced_models'][-1]['0.bias'][inds]
        # print(len(bias[bias > 0.05]))
        plt.plot(bias, label=str(sweep_vars[i]), c=scalarMap.to_rgba(sweep_vars_col[i]))
    plt.axvline(512,c='k', label='# Features')
    plt.xlabel('Neuron')
    plt.ylabel('Bias')

    max_k = max(batch[i]['k'] for i in range(len(batch)))
    plt.xlim([-0.1*max_k, 1.1*max_k])

    cax = fig.add_axes([0.11,0.9,0.25,0.02])
    cb = fig.colorbar(scalarMap, cax=cax, orientation='horizontal')
    if log_color:
        cax.set_title(r'$\log_{10}$' + nice_names[sweep_var])
    else:
        cax.set_title(f'{nice_names[sweep_var]}')

    return fig
    
def sfa_plot(batch, sweep_var, js, crop=[1024,512]):
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.04)

    fig = plt.figure(figsize=(13,18))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    axes = [ax1,ax2,ax3]

    for i,j in enumerate(js):
        sfa = single_feature_activations(batch[j]['log2_spaced_models'][-1], batch[j], batch[j]['setup'])

        # Sort the neurons to put the most-monosemantic first
        inds = np.argsort(-np.amax(sfa,axis=1) / (1e-10 + np.mean(sfa,axis=1)))
        sfa = sfa[inds]

        # Sort the features to put the most-monosemantic neurons first
        neuron_inds = []
        for k in range(sfa.shape[1]): # Loop over features
            neuron_ind = np.argmax(sfa[:,k]) # Find the neuron this feature activates most-strongly.
            neuron_inds.append(neuron_ind)
        inds = np.argsort(neuron_inds) # Sort the neuron indices
        sfa = sfa[:,inds]

        im = axes[i].imshow(sfa[:crop[0],:crop[1]],interpolation='nearest', aspect=1.8, vmin=0, vmax=1.02)
        axes[i].annotate(f'{nice_names[sweep_var]}={round(batch[j][sweep_var],4)}', (10,20), c='white')
        axes[i].set_xlabel('Feature')
    
    cbar = fig.colorbar(im, orientation='horizontal', ax=axes, location='top', pad=0.01, aspect=40)

    ax1.set_ylabel('Neuron')
    ax2.set_yticks([])
    ax3.set_yticks([])
    return fig

def mfa_plot(batch, sweep_var, js, extras):
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.06)

    fig = plt.figure(figsize=(15,20))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    axes = [ax1,ax2,ax3]

    for i,j in enumerate(js):
        sfa = many_feature_activations(batch[j]['log2_spaced_models'][-1], batch[j], batch[j]['setup'], extras)

        # Sort the neurons to put the most-monosemantic first
        inds = np.argsort(-np.amax(sfa,axis=1) / (1e-10 + np.mean(sfa,axis=1)))
        sfa = sfa[inds]

        im = axes[i].imshow(sfa.T,interpolation='nearest')
        axes[i].annotate(f'{nice_names[sweep_var]}={batch[j][sweep_var]}', (40,40), c='white')
        axes[i].set_ylabel('Feature')
        fig.colorbar(im, orientation='vertical', ax=axes[i])

    ax3.set_xlabel('Neuron')
    return fig

def sfa_line_plot(data, extras):
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0)

    fig = plt.figure(figsize=(7,6))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    axes = [ax1,ax2,ax3]
    
    sfa = single_feature_activations(data['log2_spaced_models'][-1], data, data['setup'])

    # Sort the neurons to put the most-monosemantic first
    inds = np.argsort(-np.amax(sfa,axis=1) / (1e-10 + np.mean(sfa,axis=1)))
    sfa = sfa[inds]

    for i in range(3):
        ax = axes[i]
        ax.plot(sfa[:,extras[i]])
        ax.set_ylabel(f'Activation\n Feature {extras[i]}')

    ax3.set_xlabel('Neuron')
    return fig

def plot_mono_sweep(batch, sweep_var):
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0)

    fig = plt.figure(figsize=(10,9))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    axes = [ax1,ax2]

    plot_data = []
    for j in range(len(batch)):
        sfa = single_feature_activations(batch[j]['log2_spaced_models'][-1], batch[j], batch[j]['setup'])
        frac_mono = np.amax(sfa,axis=1) / (1e-10 + np.sum(sfa,axis=1))
        plot_data.append((batch[j][sweep_var],sum(frac_mono > r_threshold), sum(frac_mono > r_threshold)/len(frac_mono)))

    plot_data = np.array(plot_data)

    ax1.plot(plot_data[:,0],plot_data[:,2])
    if sweep_var == 'k':
        ax1.axvline(512, linestyle=':', c='r')
    ax1.axhline(1, linestyle=':', c='k')
    ax1.set_ylabel('# Mono Neurons /\n # Neurons')
    ax1.set_xticks([])
        
    ax2.plot(plot_data[:,0],plot_data[:,1]/512)
    if sweep_var == 'k':
        plt.axvline(512, linestyle=':', c='r')
    ax2.axhline(1, linestyle=':', c='k')
    ax2.set_ylabel('# Mono Neurons /\n # Features')
    ax2.set_xlabel(nice_names[sweep_var])

    return fig


def plot_bias_vs_freq_activated(batch, color='#947EB0', step_size=0.04):
    d = batch['log2_spaced_models'][-1]
    sfa = single_feature_activations(d, batch, batch['setup'])
    # Sort the neurons to put the most-monosemantic first
    inds = np.argsort(-np.amax(sfa, axis=1) / (1e-10 + np.mean(sfa, axis=1)))
    sorted_bias = d['0.bias'].detach().numpy()[inds]

    unit_sfa = np.array(sfa > 0.0, dtype='int')
    sorted_freq = np.sum(unit_sfa, axis=-1)[inds]

    gs = GridSpec(2, 1, height_ratios=[0.5, 1], hspace=0)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    n, _, _ = ax1.hist([sorted_bias],
                       bins=np.arange(np.rint(np.min(sorted_bias)), np.max(sorted_bias) + step_size, step_size),
                       color=color, histtype='step', lw=4)
    ax1.plot([0, 0], [0.0, np.max(n) + 0.1 * np.max(n)], 'k--')
    ax1.set_xlim(np.min(sorted_bias) - 0.1*np.abs(np.min(sorted_bias)),
                 np.max(sorted_bias) - 0.1*np.abs(np.max(sorted_bias)))
    ax1.set_ylim(0.0, np.max(n) + 0.1 * np.max(n))
    # ax1.set_yscale('log')
    ax1.set_ylabel('freq.')
    ax1.set_xticks([])
    ax1.get_shared_x_axes().join(ax1, ax2)

    ax2.scatter(sorted_bias, sorted_freq, color=color)
    ax2.plot([0, 0], [0.0, np.max(sorted_freq) + 0.1 * np.max(sorted_freq)], 'k--')
    # ax2.set_xlim(np.min(sorted_bias), 0.4)
    ax2.set_ylim(-0.01 * np.max(sorted_freq), np.max(sorted_freq) + 0.1 * np.max(sorted_freq))
    # ax2.set_yscale('log')
    ax2.set_xlabel('bias')
    ax2.set_ylabel('freq. activated')

    return fig


def plot_dead_mono_poly(batch, color='#947EB0'):
    d = batch['log2_spaced_models'][-1]
    sfa = single_feature_activations(d, batch, batch['setup'])
    print(np.mean(sfa))
    unit_sfa = np.array(sfa > 0.0, dtype='int')
    counts = Counter(np.sum(unit_sfa, axis=-1))
    # print(counts)
    c = [np.sum([counts[k] for k in counts if k not in [0, 1]]), counts[1], counts[0]]
    # print(c)

    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    bins = range(0, batch['N'] + 8, 8)
    ax1.hist(np.sum(unit_sfa, axis=-1), bins=bins, color=color)
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Activating Features')
    ax1.set_ylabel('Neuron Frequency')
    ax1.set_xlim(-1, batch['N'] + 1)

    y_pos = range(0, 3, 1)
    ax2.barh(y_pos, c, color=color)
    ax2.set_xlabel('Number of Neurons')
    ax2.set_yticks(y_pos, ['poly.', 'mono.', 'dead'])

    return fig, c


def plot_number_activating_neurons_and_features(batch, save_path, color='#947EB0'):
    sfa = single_feature_activations(batch['log2_spaced_models'][-1], batch, batch['setup'])

    # print(sfa.shape)

    n_activating_features = np.sum(np.array(sfa > 0.0, dtype='int'), axis=-1)

    # print(n_activating_features.shape)

    with open(os.path.join(save_path, 'neurons_to_features_report.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        header = ['neuron_id', 'n_activating_features', 'activating_features']
        writer.writerow(header)
        for i in range(sfa.shape[0]):
            row = [
                f'{i+1: 04d}',
                n_activating_features[i],
                [f'{j+1: 04d}' for j in np.where(sfa[i,:] > 0.0)[0]]
            ]
            writer.writerow(row)

    n_activating_neurons = np.sum(np.array(sfa > 0.0, dtype='int'), axis=0)

    with open(os.path.join(save_path, 'features_to_neurons_report.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        header = ['feature_id', 'n_activating_neurons', 'activating_neurons']
        writer.writerow(header)
        for i in range(sfa.shape[1]):
            row = [
                f'{i+1: 04d}',
                n_activating_neurons[i],
                [f'{j+1: 04d}' for j in np.where(sfa[:,i] > 0.0)[0]]
            ]
            writer.writerow(row)

    # Sort the neurons to put the most-monosemantic first
    inds = np.argsort(-np.amax(sfa, axis=1) / (1e-10 + np.mean(sfa, axis=1)))
    sfa = sfa[inds]

    # Sort the features to put the most-monosemantic neurons first
    neuron_inds = []
    for k in range(sfa.shape[1]):  # Loop over features
        neuron_ind = np.argmax(sfa[:, k])  # Find the neuron this feature activates most-strongly.
        neuron_inds.append(neuron_ind)
    inds = np.argsort(neuron_inds)  # Sort the neuron indices
    sfa = sfa[:, inds]

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_im = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    nullfmt = NullFormatter()  # no labels

    x_pos = range(0, batch['N'], 1)
    y_pos = range(0, batch['k'], 1)

    fig = plt.figure(figsize=(20, 40))

    axIm = plt.axes(rect_im)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axIm.imshow(sfa > 0.0, interpolation='nearest', vmin=0, vmax=1.02)
    axIm.set_xlabel('Feature')
    axIm.set_ylabel('Neuron')

    axHistx.bar(x_pos, np.sum(np.array(sfa > 0.0, dtype='int'), axis=0), width=1, color=color)
    # axHistx.set_yscale('log')
    axHistx.set_xlim(0, 512)
    # axHistx.set_ylim(0,1024)
    axHistx.set_ylabel('Number of Activating Features')

    axHisty.barh(y_pos, np.sum(np.array(sfa > 0.0, dtype='int'), axis=-1), height=1, color=color)

    axHisty.set_xscale('log')
    axHisty.set_ylim(0, 1024)
    axHisty.set_xlim(xmax=512)
    axHisty.set_xlabel('Number of Activating Neurons')
    axHisty.invert_yaxis()

    return fig


def plot_mech_interpretability(batch, save_path, step_size=.05,
                               clist=('tab:blue', 'tab:orange', 'tab:green', '#8B8BAE', '#DB2763')):

    for d in ['neg_pos_decomp', 'mono_poly_decomp', 'decomp_comparison']:
        p = os.path.join(save_path, d)
        try:
            os.makedirs(p)
        except FileExistsError:
            pass

    try:
        fixed_embedder = batch['setup']['fixed_embedder']
    except KeyError:
        # diabetes task
        pass

    d = batch['log2_spaced_models'][-1]

    model = model_builder(d['2.weight'].shape[0], d['0.weight'].shape[1], d['0.weight'].shape[0], batch['nonlinearity'])
    model.load_state_dict(batch['log2_spaced_models'][-1])
    model.to('cpu')

    where_pos_bias = np.where(d['0.bias'].detach().numpy() >= 0.0)[0]
    neg_d = deepcopy(batch['log2_spaced_models'][-1])
    neg_d['0.weight'][where_pos_bias, :] *= 0
    neg_d['0.bias'][where_pos_bias] *= 0
    neg_model = model_builder(neg_d['2.weight'].shape[0], neg_d['0.weight'].shape[1],
                              neg_d['0.weight'].shape[0], batch['nonlinearity'])
    neg_model.load_state_dict(neg_d)
    neg_model.to('cpu')

    where_neg_bias = np.where(d['0.bias'].detach().numpy() < 0.0)[0]
    pos_d = deepcopy(batch['log2_spaced_models'][-1])
    pos_d['0.weight'][where_neg_bias, :] *= 0
    pos_d['0.bias'][where_neg_bias] *= 0
    pos_model = model_builder(d['2.weight'].shape[0], d['0.weight'].shape[1],
                              d['0.weight'].shape[0], batch['nonlinearity'])
    pos_model.load_state_dict(pos_d)
    pos_model.to('cpu')

    sfa = single_feature_activations(d, batch, batch['setup'])
    unit_sfa = np.array(sfa > 0.0, dtype='int')

    where_poly = np.where(np.sum(unit_sfa, axis=-1) > 1)[0]
    mono_d = deepcopy(batch['log2_spaced_models'][-1])
    mono_d['0.weight'][where_poly, :] *= 0
    mono_d['0.bias'][where_poly] *= 0
    mono_model = model_builder(d['2.weight'].shape[0], d['0.weight'].shape[1],
                               d['0.weight'].shape[0], batch['nonlinearity'])
    mono_model.load_state_dict(mono_d)
    mono_model.to('cpu')

    where_mono = np.where(np.sum(unit_sfa, axis=-1) == 1)[0]
    poly_d = deepcopy(batch['log2_spaced_models'][-1])
    poly_d['0.weight'][where_mono, :] *= 0
    poly_d['0.bias'][where_mono] *= 0
    poly_model = model_builder(d['2.weight'].shape[0], d['0.weight'].shape[1],
                               d['0.weight'].shape[0], batch['nonlinearity'])
    poly_model.load_state_dict(poly_d)
    poly_model.to('cpu')

    full_os = []
    neg_os = []
    pos_os = []
    mono_os = []
    poly_os = []

    # diabetes task
    if 'diabetes' in save_path:
        from sklearn.datasets import load_diabetes
        from sklearn.preprocessing import StandardScaler, minmax_scale
        diabetes = load_diabetes(scaled=False)
        inputs, vectors = diabetes.data, diabetes.target
        scaler = StandardScaler().fit(inputs)
        inputs = scaler.transform(inputs)
        vectors = minmax_scale(vectors)

        min_i, max_i = np.min(inputs), np.max(inputs)

        for a in np.arange(min_i - step_size, max_i + 2*step_size, step_size):
            ins = torch.eye(batch['N']) * a
            full_os.append(model.forward(ins).T.detach().numpy()[0])
            neg_os.append(neg_model.forward(ins).T.detach().numpy()[0])
            pos_os.append(pos_model.forward(ins).T.detach().numpy()[0])
            mono_os.append(mono_model.forward(ins).T.detach().numpy()[0])
            poly_os.append(poly_model.forward(ins).T.detach().numpy()[0])

        full_os = np.array(full_os)
        neg_os = np.array(neg_os)
        pos_os = np.array(pos_os)
        mono_os = np.array(mono_os)
        poly_os = np.array(poly_os)

        for i in range(batch['N']):
            # f = []
            # n = []
            # p = []
            # mono = []
            # poly = []
            # for j in range(len(full_os)):
            #     f.append(full_os[j][0,i])
            #     n.append(neg_os[j][0,i])
            #     p.append(pos_os[j][0,i])
            #     mono.append(mono_os[j][0,i])
            #     poly.append(poly_os[j][0,i])
            fig = plt.figure(figsize=(10, 10))
            plt.scatter(inputs[:, i], vectors, color='#7D7C7A', alpha=0.25)
            plt.plot(np.arange(min_i - step_size, max_i + 2 * step_size, step_size), full_os[:,i], '-',
                     label='Full Model', color=clist[0], lw=6)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), neg_os[:,i], '-',
                     label='Neg. Bias Model', color=clist[1], lw=4)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), pos_os[:,i], '-',
                     label='Pos. Bias Model', color=clist[2], lw=4)
            # plt.plot([0, 1], [0, 1], 'k--', label='Truth')
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            plt.xlabel('Input Feature Amplitude')
            plt.ylabel('Output Amplitude')
            plt.legend()

            fig.tight_layout()
            fig.savefig(
                os.path.join(save_path, 'neg_pos_decomp', f'mech_interp_neg_pos_decomp_feature_{i + 1:04d}.png'))

            fig = plt.figure(figsize=(10, 10))
            plt.scatter(inputs[:, i], vectors, color='#7D7C7A', alpha=0.25)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), full_os[:,i], '-',
                     label='Full Model', color=clist[0], lw=6)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), mono_os[:,i], '-',
                     label='Mono. Model', color=clist[3], lw=4)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), poly_os[:,i], '-',
                     label='Poly. Bias Model', color=clist[4], lw=4)
            # plt.plot([0, 1], [0, 1], 'k--', label='Truth')
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            plt.xlabel('Input Feature Amplitude')
            plt.ylabel('Output Amplitude')
            plt.legend()

            fig.tight_layout()
            fig.savefig(
                os.path.join(save_path, 'mono_poly_decomp', f'mech_interp_mono_poly_decomp_feature_{i + 1:04d}.png'))

            fig = plt.figure(figsize=(10, 10))
            plt.scatter(inputs[:, i], vectors, color='#7D7C7A', alpha=0.25)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), full_os[:,i], '-',
                     label='Full Model', color=clist[0], lw=6)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), neg_os[:,i], '-',
                     label='Neg. Bias Model', color=clist[1], lw=4)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), pos_os[:,i], '-',
                     label='Pos. Bias Model', color=clist[2], lw=4)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), mono_os[:,i], '-',
                     label='Mono. Model', color=clist[3], lw=4)
            plt.plot(np.arange(min_i - step_size, max_i + 2*step_size, step_size), poly_os[:,i], '-',
                     label='Poly. Model', color=clist[4], lw=4)
            # plt.plot([0, 1], [0, 1], 'k--', label='Truth')
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            plt.xlabel('Input Feature Amplitude')
            plt.ylabel('Output Amplitude')
            plt.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'decomp_comparison',
                                     f'mech_interp_decomp_comparison_feature_{i + 1:04d}.png'))
            plt.close('all')
    else:

        for a in np.arange(0, 1 + step_size, step_size):
            vs = torch.eye(batch['N']) * a
            ins = torch.matmul(vs, fixed_embedder.T)

            full_os.append(model.forward(ins).T.detach().numpy())
            neg_os.append(neg_model.forward(ins).T.detach().numpy())
            pos_os.append(pos_model.forward(ins).T.detach().numpy())
            mono_os.append(mono_model.forward(ins).T.detach().numpy())
            poly_os.append(poly_model.forward(ins).T.detach().numpy())

        for i in range(batch['N']):
            f = []
            n = []
            p = []
            mono = []
            poly = []
            for j in range(len(full_os)):
                f.append(full_os[j][i,i])
                n.append(neg_os[j][i,i])
                p.append(pos_os[j][i,i])
                mono.append(mono_os[j][i,i])
                poly.append(poly_os[j][i,i])

            fig = plt.figure(figsize=(10, 10))
            plt.plot(np.arange(0, 1 + step_size, step_size), f, '-', label='Full Model', color=clist[0])
            plt.plot(np.arange(0, 1 + step_size, step_size), n, '-', label='Neg. Bias Model', color=clist[1])
            plt.plot(np.arange(0, 1 + step_size, step_size), p, '-', label='Pos. Bias Model', color=clist[2])
            plt.plot([0, 1], [0, 1], 'k--', label='Truth')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('Input Feature Amplitude')
            plt.ylabel('Output Amplitude')
            plt.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'neg_pos_decomp', f'mech_interp_neg_pos_decomp_feature_{i+1:04d}.png'))

            fig = plt.figure(figsize=(10, 10))
            plt.plot(np.arange(0, 1 + step_size, step_size), f, '-', label='Full Model', color=clist[0])
            plt.plot(np.arange(0, 1 + step_size, step_size), mono, '-', label='Mono. Model', color=clist[3])
            plt.plot(np.arange(0, 1 + step_size, step_size), poly, '-', label='Poly. Bias Model', color=clist[4])
            plt.plot([0, 1], [0, 1], 'k--', label='Truth')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('Input Feature Amplitude')
            plt.ylabel('Output Amplitude')
            plt.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'mono_poly_decomp', f'mech_interp_mono_poly_decomp_feature_{i+1:04d}.png'))

            fig = plt.figure(figsize=(10, 10))
            plt.plot(np.arange(0, 1 + step_size, step_size), f, '-', label='Full Model', color=clist[0])
            plt.plot(np.arange(0, 1 + step_size, step_size), n, '-', label='Neg. Bias Model', color=clist[1])
            plt.plot(np.arange(0, 1 + step_size, step_size), p, '-', label='Pos. Bias Model', color=clist[2])
            plt.plot(np.arange(0, 1 + step_size, step_size), mono, '-', label='Mono. Model', color=clist[3])
            plt.plot(np.arange(0, 1 + step_size, step_size), poly, '-', label='Poly. Model', color=clist[4])
            plt.plot([0, 1], [0, 1], 'k--', label='Truth')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('Input Feature Amplitude')
            plt.ylabel('Output Amplitude')
            plt.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'decomp_comparison',
                                     f'mech_interp_decomp_comparison_feature_{i+1:04d}.png'))
            plt.close('all')
