import os
import os.path as osp
import collections
import tarfile
import zipfile
import torch.utils.data
from dataloader.read_graph_datasets import read_graph_data
import random
import numpy as np
from six.moves import urllib
import errno
from utils.compute_wl_kernel import compute_reduce_wl_kernel, compute_full_wl_kernel
# from utils.compute_fgsd_features import compute_reduce_fgsd_features
import time
import logging


def extract_tar(path, folder, mode='r:gz', log=True):
    maybe_log(path, log)
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def extract_zip(path, folder, log=True):
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder, log=True):
    if log:
        print('Downloading', url)

    makedirs(folder)

    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def to_list(x):
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x


def files_exist(files):
    return all([osp.exists(f) for f in files])


class TUDataset(torch.utils.data.Dataset):
    url = 'https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets'

    def __init__(self, data_path, name, shuffle=True, compute_graph_kernel_features=False, wl_num_iter=5, wl_node_labels='degree', compute_fgsd_features=False):

        self.name = name
        self.compute_graph_kernel_features = compute_graph_kernel_features
        self.wl_num_iter = wl_num_iter
        self.wl_node_labels = wl_node_labels

        self.root = osp.expanduser(osp.normpath(data_path))
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        if self.name != 'ALL' and self.name != 'QM8' and self.name != 'NCI_FULL':
            self._download()
        self._process()
        logging.info('Loading save graph list from: ' + os.path.abspath(self.processed_paths[0]))
        self.graph_list = torch.load(self.processed_paths[0])
        if shuffle:
            logging.info('Shuffling the dataset...')
            random.shuffle(self.graph_list)
        if self.name == 'QM8':
            self.num_graph_labels = len(self.graph_list[0]['graph_label'])
        elif self.name != 'ALL':
            self.num_graph_labels = max(G['graph_label'].item() for G in self.graph_list) + 1
        else:
            self.num_graph_labels = None

    def __getitem__(self, index):
        # return self.graph_list[index]
        return [self.graph_list[i] for i in index]

    def __len__(self):
        return len(self.graph_list)

    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def raw_paths(self):
        files = to_list(self.raw_file_names())
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self):
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def _download(self):
        if files_exist(self.raw_paths):
            return
        makedirs(self.raw_dir)
        self.download()

    @property
    def num_features(self):
        return self.graph_list[0]['node_feature_matrix'].shape[-1]

    @property
    def wl_kernel_feature_dim(self):
        return self.graph_list[0]['WL_kernel_features'].shape[-1]

    @property
    def num_classes(self):
        return self.num_graph_labels

    def _process(self):
        if files_exist(self.processed_paths):
            return
        print('Processing...')
        makedirs(self.processed_dir)
        self.process()
        print('Done!')

    def download(self):
        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        if self.name == 'ALL':
            if os.path.exists('data/ALL/processed/graph_list.pt'):
                self.graph_list = torch.load('data/ALL/processed/graph_list.pt')
            else:
                self.graph_list = []
                for i in range(len(graph_dataset_names)):
                    logging.info('Reading graph dataset: ' + graph_dataset_names[i])
                    dataset_graph_list, _ = read_graph_data('data/' + graph_dataset_names[i], graph_dataset_names[i])
                    self.graph_list = self.graph_list + dataset_graph_list
                    logging.info('Done')
                self.num_graph_labels = None
                torch.save(self.graph_list, 'data/ALL/processed/graph_list.pt')
        elif self.name == 'QM8':
            from dataloader.read_qm8_dataset import read_qm8_data
            self.graph_list, self.num_graph_labels = read_qm8_data(self.raw_dir, self.name)
        elif self.name == 'NCI_FULL':
            from dataloader.read_nci_full_dataset import read_nci_full_data
            self.graph_list, self.num_graph_labels = read_nci_full_data(self.raw_dir, self.name)
        else:
            logging.info('Reading graph dataset from: ' + os.path.abspath(self.raw_dir))
            self.graph_list, self.num_graph_labels = read_graph_data(self.raw_dir, self.name)

        if self.compute_graph_kernel_features:
            feature_matrix = compute_full_wl_kernel(self.graph_list, num_iter=self.wl_num_iter, type_node_labels=self.wl_node_labels)
            for i in range(len(self.graph_list)):
                if (i + 1) % 1000 == 0:
                    logging.info('wl features loaded in num graphs so far: ' + str(i + 1))
                self.graph_list[i]['WL_kernel_features'] = feature_matrix[i]

        logging.info('Saving the process data at: ' + os.path.abspath(self.processed_paths[0]))
        t = time.time()
        torch.save(self.graph_list, self.processed_paths[0])
        logging.info('Time Taken: ' + str(time.time() - t))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
