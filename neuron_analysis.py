import csv
import argparse

from plot_helper import *
from copy import deepcopy

parser = argparse.ArgumentParser()
# parser.add_argument("neuron", type=int)
parser.add_argument("model", type=str)
parser.add_argument("k", type=int)
parser.add_argument("log2_batch_size", type=int)
parser.add_argument("log2_training_steps", type=int)
parser.add_argument("sample_kind")
parser.add_argument("init_bias", type=float)
parser.add_argument("nonlinearity")
parser.add_argument("task")
parser.add_argument("decay", type=float)
parser.add_argument("eps", type=float)
parser.add_argument("m", type=int)
parser.add_argument("N", type=int)
parser.add_argument("reg", type=float)
parser.add_argument("step_size", type=float)
parser.add_argument("--lr", type=float)

args = parser.parse_args()

# neuron = args.neuron
model = args.model
k = args.k
log2_batch_size = args.log2_batch_size
log2_training_steps = args.log2_training_steps
sample_kind = args.sample_kind
init_bias = args.init_bias
nonlinearity = args.nonlinearity
task = args.task
decay = args.decay
eps = args.eps
m = args.m
N = args.N
reg = args.reg
step_size = args.step_size
lr = args.lr

root = f'./my_plots/{model}_{task}_{nonlinearity}_k_{k}_batch_{log2_batch_size}_steps_{log2_training_steps}_sample_{sample_kind}_init_bias_{init_bias}_decay_{decay}_eps_{eps}_m_{m}_N_{N}_reg_{reg}/'
try:
    os.makedirs(root)
except FileExistsError:
    pass

# Load and process data
names = list([
    f"./my_models/{model}_{task}_{nonlinearity}_k_{k}_batch_{log2_batch_size}_steps_{log2_training_steps}_learning_rate_{lr}_sample_{sample_kind}_init_bias_{init_bias}_decay_{decay}_eps_{eps}_m_{m}_N_{N}_reg_{reg}.pt"
])

ReLU_equal_lr_sweep = []
for n in names:
    try:
        ReLU_equal_lr_sweep.append(torch.load(n, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(n,'not found')

for batch in ReLU_equal_lr_sweep:
    p = os.path.join(root, f'learning_rate_{batch["learning_rate"]}', 'neurons')
    try:
        os.makedirs(p)
    except FileExistsError:
        pass

    with open(os.path.join(p, 'activation_count.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['neuron_id', 'n_activating_single', 'n_activating_all'])
        for neuron in range(1, k+1):
            fixed_embedder = batch['setup']['fixed_embedder']
            d = batch['log2_spaced_models'][-1]
            # print(d)
            model = model_builder(d['2.weight'].shape[0], d['0.weight'].shape[1], d['0.weight'].shape[0], batch['nonlinearity'])
            masked_weights = torch.zeros_like(d['0.weight'])
            masked_weights[neuron-1, :] = d['0.weight'][neuron-1, :]
            # print(masked_weights[neuron-1])
            masked_bias = torch.zeros_like(d['0.bias'])
            masked_bias[neuron-1] = d['0.bias'][neuron-1]
            # print(np.unique(masked_bias))
            masked_d = deepcopy(batch['log2_spaced_models'][-1])
            masked_d['0.weight'] = masked_weights
            masked_d['0.bias'] = masked_bias
            model.load_state_dict(masked_d)
            model.to('cpu')

            print(f'### neuron_{neuron:04}: Single feature active ###')
            out = []
            sls = 0
            for a in np.arange(0, 1 + step_size, step_size):
                vs = torch.eye(batch['N']) * a
                ins = torch.matmul(vs, fixed_embedder.T)
                out.append(model.forward(ins).T.detach().numpy())

            fig = plt.figure(figsize=(10,10))
            for i in range(batch['N']):
                f = []
                for j in range(len(out)):
                    f.append(out[j][i, i])
                color = matplotlib.cm.rainbow(i)
                if np.sum(f) > 0.0:
                    # print(f)
                    print(f'feature_{i+1:03} ' +
                          f'{np.max(f)}')
                    sls += 1
                    plt.plot(np.arange(0, 1 + step_size, step_size), f, '-', label=f'feature_{i+1:04d}', color=color)
            # plt.yscale('log')
            plt.xlabel('Input Feature Amplitude')
            plt.ylabel('Output Amplitude')
            if sls <= 10:
                plt.legend()

            fig.tight_layout()
            plt.savefig(os.path.join(p, f'single_feature_activation_neuron_{neuron:04d}.png'))
            plt.close('all')

            print(f'Number of activating features = {sls}')

            print(f'### neuron_{neuron:04}: All features equally active ###')
            out = []
            als = 0
            for a in np.arange(0, 1 + step_size, step_size):
                vs = torch.ones(batch['N'], batch['N']) * a
                ins = torch.matmul(vs, fixed_embedder.T)
                out.append(model.forward(ins).T.detach().numpy())

            fig = plt.figure(figsize=(10,10))
            for i in range(batch['N']):
                f = []
                for j in range(len(out)):
                    f.append(out[j][i, i])
                color = matplotlib.cm.rainbow(i)
                if np.sum(f) > 0.0:
                    # print(f)
                    print(f'feature_{i+1:03} ' +
                          f'{np.max(f)}')
                    als += 1
                    plt.plot(np.arange(0, 1 + step_size, step_size), f, '-', label=f'feature_{i+1:04d}', color=color)
            # plt.yscale('log')
            plt.xlabel('Input Feature Amplitude')
            plt.ylabel('Output Amplitude')
            if als <= 10:
                plt.legend()

            fig.tight_layout()
            plt.savefig(os.path.join(p, f'all_feature_activation_neuron_{neuron:04d}.png'))
            plt.close('all')

            print(f'Number of activating features = {als}')
            writer.writerow([f'{neuron:04}', sls, als])

