import os
import sys
import subprocess
from itertools import islice
import shutil
import errno
sys.path.extend('..')
from config.params_universal_graph_embedding_model import params_list, hyperparams_grid
import uuid
import argparse


def contruct_py_cmd_args(params_dict):
    cmd_args = ''
    for key, value in params_dict.items():
        cmd_args = cmd_args + '--' + str(key) + ' ' + str(value) + ' '
    return cmd_args


if __name__ == '__main__':

    experiment_id = str(uuid.uuid4())
    dest_path = os.path.join('./hyperopt_runs/', experiment_id)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    print('Copying files...')
    src_list = ['./train', './utils', './torch_dgl', './dataloader', './config']
    dest_list = [os.path.join(dest_path, 'train'),
                 os.path.join(dest_path, 'utils'),
                 os.path.join(dest_path, 'torch_dgl'),
                 os.path.join(dest_path, 'dataloader'),
                 os.path.join(dest_path, 'config')]

    for dataset in hyperparams_grid['dataset_name']:
        src_list = src_list + ['./data/' + dataset]
        dest_list = dest_list + [dest_path + '/data/' + dataset]
    src_list = src_list + ['./data/prime_numbers_list_v2.npy']
    dest_list = dest_list + [dest_path + '/data/prime_numbers_list_v2.npy']

    for src, dest in zip(src_list, dest_list):
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        try:
            shutil.copytree(src, dest)
        except NotADirectoryError:
            shutil.copy(src, dest)

    os.chdir(dest_path)
    print('Done...')

    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument('--train_filename', type=str, default='train_universal_graph_embedding.py')
    parser.add_argument('--max_workers', type=int, default=2)
    args, unknown = parser.parse_known_args()

    for arg, value in sorted(vars(args).items()):
        print("Hyperparameter: %s: %r", arg, value)

    train_filename = args.train_filename
    commands = []
    for i in range(len(params_list)):
        py_cmd_args = contruct_py_cmd_args(params_list[i])
        run_cmd = 'python' + ' ' + os.path.join('train/', train_filename) + ' ' + py_cmd_args  # + ' &'
        commands.append(run_cmd)

    max_workers = args.max_workers
    processes = (subprocess.Popen(cmd, shell=True) for cmd in commands)
    running_processes = list(islice(processes, max_workers))  # start new processes
    while running_processes:
        for i, process in enumerate(running_processes):
            if process.poll() is not None:  # the process has finished
                running_processes[i] = next(processes, None)  # start new process
                if running_processes[i] is None:  # no new processes
                    del running_processes[i]
                    break
