from tsalib import dim_vars
from sklearn.model_selection import ParameterSampler


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


B, N, E = dim_vars('Batch_Size Graph_Nodes Edge_List')
D, H, K, C = dim_vars('Node_Input_Features Node_Hidden_Features Graph_Kernel_Dim Num_Classes')

hyperparams_grid = {
    'model_name': ['universal-graph-embedding'],
    'dataset_name': ['DD'],
    'save_steps': [200],
    'run_on_comet': [True],
    'gpu_device': [0],

    'hidden_dim': [16, 32, 64],
    'num_gconv_layers': [5, 7],
    'num_gfc_layers': [2, 4],
    'batch_size': [128, 64, 32],
    'drop_prob': [0, 0.2],
    'num_epochs': [3000]
}


gen_params_set = 1
for key, val in hyperparams_grid.items():
    gen_params_set = gen_params_set * len(val)

params_list = list(ParameterSampler(hyperparams_grid, n_iter=gen_params_set))
