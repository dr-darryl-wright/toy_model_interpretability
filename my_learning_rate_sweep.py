import os
import csv
import argparse
from plot_helper import *

parser = argparse.ArgumentParser()
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
parser.add_argument("--lrs", nargs="+", type=float, default=[0.001, 0.003, 0.005, 0.007, 0.01, 0.03])

args = parser.parse_args()

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
lrs = args.lrs

print(lrs)

root = f'./my_plots/{model}_{task}_{nonlinearity}_k_{k}_batch_{log2_batch_size}_steps_{log2_training_steps}_sample_{sample_kind}_init_bias_{init_bias}_decay_{decay}_eps_{eps}_m_{m}_N_{N}_reg_{reg}/'
try:
    os.makedirs(root)
except FileExistsError:
    pass

# Load and process data
names = list([
    f"./my_models/{model}_{task}_{nonlinearity}_k_{k}_batch_{log2_batch_size}_steps_{log2_training_steps}_learning_rate_{lr}_sample_{sample_kind}_init_bias_{init_bias}_decay_{decay}_eps_{eps}_m_{m}_N_{N}_reg_{reg}.pt"
    for lr in lrs
])
ReLU_equal_lr_sweep = []
for n in names:
    try:
        ReLU_equal_lr_sweep.append(torch.load(n, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(n,'not found')

fig = training_plot(ReLU_equal_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig(os.path.join(root, 'lr_sweep_training_plot.png'))

if len(lrs) >= 6:
    fig = sfa_plot(ReLU_equal_lr_sweep, 'learning_rate', [0,2,5])
    fig.tight_layout()
    fig.savefig(os.path.join(root, 'lr_sweep_sfa_plot.png'))

fig = plot_bias(ReLU_equal_lr_sweep, 'learning_rate', log_color=True)
fig.tight_layout()
fig.savefig(os.path.join(root, 'lr_sweep_bias_plot.png'))

# for batch in ReLU_equal_lr_sweep:
#     p = os.path.join(root, f'learning_rate_{batch["learning_rate"]}')
#     try:
#         os.makedirs(p)
#     except FileExistsError:
#         pass
#     plot_mech_interpretability(batch, p)

with open(os.path.join(root, 'in_domain_report.csv'), 'w') as in_domain_csvfile, \
     open(os.path.join(root, 'out_of_domain_report.csv'), 'w') as out_of_domain_csvfile:
    in_domain_writer = csv.writer(in_domain_csvfile)
    out_of_domain_writer = csv.writer(out_of_domain_csvfile)
    header = ['lr', 'n_mono', 'n_poly', 'n_dead']
    in_domain_writer.writerow(header)
    out_of_domain_writer.writerow(header)
    for batch in ReLU_equal_lr_sweep:
        print(batch["learning_rate"])
        p = os.path.join(root, f'learning_rate_{batch["learning_rate"]}')
        try:
            os.makedirs(p)
        except FileExistsError:
            pass

        fig = plot_bias_vs_freq_activated(batch)
        fig.tight_layout()
        fig.savefig(os.path.join(p, 'bias_vs_n_activating_neurons_plot.png'))
#
        fig, counts = plot_dead_mono_poly(batch, color='#947EB0')
        fig.tight_layout()
        fig.savefig(os.path.join(p, 'dead_mono_poly_plot.png'))
#
        print(counts)
        row = [batch['learning_rate'], counts[0][0], counts[0][1], counts[0][2]]
        in_domain_writer.writerow(row)
        row = [batch['learning_rate'], counts[1][0], counts[1][1], counts[1][2]]
        out_of_domain_writer.writerow(row)
#
#         fig = plot_number_activating_neurons_and_features(batch, save_path=p, color='#947EB0')
#         fig.tight_layout()
#         fig.savefig(os.path.join(p, 'number_activating_neurons_and_features_plot.png'))
#
        # plot_mech_interpretability(batch, p)
