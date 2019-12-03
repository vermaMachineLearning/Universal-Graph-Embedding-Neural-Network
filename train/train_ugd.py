import logging
import os
import sys
sys.path.extend('..')
import numpy as np
import time
import random
import argparse
from comet_ml import Experiment
# import matlab.engine
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from dataloader.graph_datasets import TUDataset
from dataloader.graph_dataloader import DataLoader
from utils.utils import save_model, load_model, NoamLR, load_partial_model_state
from torch_dgl.models.model_universal_graph_embedding import UniversalGraphEmbedder
from config.utils_comet import API_KEY
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score, precision_score, recall_score, accuracy_score
from scipy import sparse
from sklearn import random_projection, svm
from tensorboardX import SummaryWriter
import shutil
from sklearn.preprocessing import normalize


def batch_prep_input(batch, batch_graph_list, dataset_name=None):

    feature_matrix = []
    idx_graphid = []
    for i, G in enumerate(batch_graph_list):
        feature_matrix.append(G['WL_kernel_features'])
        idx_graphid.append(G['graph_id'])

    idx_graphid = np.array(idx_graphid, dtype=np.int64)
    batch_feature_matrix = sparse.vstack(feature_matrix)
    batch_kernel_matrix = batch_feature_matrix.dot(batch_feature_matrix.transpose())
    batch_kernel_matrix = np.array(batch_kernel_matrix.todense())
    batch['wl_kernel_matrix'] = torch.FloatTensor(batch_kernel_matrix)

    if args.use_fgsd_features:

        batch_fgsd_kernel_matrix = fgsd_kernel[idx_graphid]
        batch_fgsd_kernel_matrix = batch_fgsd_kernel_matrix[:, idx_graphid]
        batch['fgsd_kernel_matrix'] = torch.FloatTensor(batch_fgsd_kernel_matrix)

    if args.use_spk_features:
        batch_spk_kernel_matrix = spk_kernel[idx_graphid]
        batch_spk_kernel_matrix = batch_spk_kernel_matrix[:, idx_graphid]
        batch['spk_kernel_matrix'] = torch.FloatTensor(batch_spk_kernel_matrix)

    graph_labels = batch['graph_labels'].data.cpu().numpy()
    class_kernel_matrix = np.equal(graph_labels, graph_labels[:, np.newaxis]).astype(int)
    batch['class_kernel_matrix'] = torch.FloatTensor(class_kernel_matrix)

    return batch


def eval_model(eval_loader):

    model.eval()
    num_samples = len(eval_loader.dataset)
    total_loss = 0
    total_class_loss = 0
    total_kernel_loss = 0
    total_fgsd_kernel_loss = 0
    total_spk_kernel_loss = 0
    total_adj_reconst_loss = 0
    correct = 0
    num_batches = 0
    X_eval = []
    Y_eval = []

    with torch.no_grad():
        for idx_batch, batch in enumerate(eval_loader):

            X = batch['node_feature_matrix'].to(device)
            A = batch['adjacency_matrix'].to(device)
            # A_mask = batch['adjacency_mask'].to(device)
            A_mask = None
            batch_sample_matrix = batch['batch_sample_matrix'].to(device)
            graph_labels = batch['graph_labels'].to(device)
            kernel_true = batch['wl_kernel_matrix'].to(device)

            class_kernel_true = None
            spk_kernel_true = None
            fgsd_kernel_true = None
            if args.use_adaptive_kernel_loss:
                class_kernel_true = batch['class_kernel_matrix'].to(device)
            if args.use_spk_features:
                spk_kernel_true = batch['spk_kernel_matrix'].to(device)
            if args.use_fgsd_features:
                fgsd_kernel_true = batch['fgsd_kernel_matrix'].to(device)

            logits, kernel_pred, fgsd_kernel_pred, spk_kernel_pred, graph_emb, A_pred = model(X, A, A_mask, batch_sample_matrix)
            loss, loss_class, loss_kernel, loss_fgsd_kernel, loss_spk_kernel, loss_adj_reconst = compute_loss(logits, graph_labels, kernel_pred, kernel_true, fgsd_kernel_pred, fgsd_kernel_true, spk_kernel_pred, spk_kernel_true, class_kernel_true, A_pred, A)

            pred = logits.max(dim=1)[1]
            correct += pred.eq(graph_labels).sum().item()

            total_loss += loss.item()
            total_class_loss += loss_class.item()
            total_kernel_loss += loss_kernel.item()
            total_fgsd_kernel_loss += loss_fgsd_kernel.item()
            total_spk_kernel_loss += loss_spk_kernel.item()
            total_adj_reconst_loss += loss_adj_reconst.item()
            num_batches = num_batches + 1
            X_eval.append(graph_emb.data.cpu().numpy())
            Y_eval.append(graph_labels.data.cpu().numpy())

    acc_val = correct / num_samples
    loss_per_sample = total_loss / num_batches
    loss_class_per_sample = total_class_loss / num_batches
    loss_kernel_per_sample = total_kernel_loss / num_batches
    loss_fgsd_kernel_per_sample = total_fgsd_kernel_loss / num_batches
    loss_spk_kernel_per_sample = total_spk_kernel_loss / num_batches
    loss_adj_reconst_per_sample = total_adj_reconst_loss / num_batches

    return loss_per_sample, loss_class_per_sample, loss_kernel_per_sample, loss_fgsd_kernel_per_sample, loss_spk_kernel_per_sample, loss_adj_reconst_per_sample, acc_val, X_eval, Y_eval


def compute_loss(logits, labels, kernel_pred, kernel_true, fgsd_kernel_pred=None, fgsd_kernel_true=None, spk_kernel_pred=None, spk_kernel_true=None, class_kernel_true=None, A_pred=None, A=None):

    loss_class = args.lambda_class_reg * F.nll_loss(logits, labels)
    loss_fgsd_kernel = torch.FloatTensor([0]).to(device)
    loss_spk_kernel = torch.FloatTensor([0]).to(device)

    if class_kernel_true is not None:
        kernel_max = torch.max(kernel_true, torch.max(fgsd_kernel_true, spk_kernel_true))
        kernel_min = torch.min(kernel_true, torch.min(fgsd_kernel_true, spk_kernel_true))
        kernel_combo = class_kernel_true * kernel_max + (1.0 - class_kernel_true) * kernel_min
        loss_kernel = args.lambda_kernel_reg * F.mse_loss(kernel_pred, kernel_combo)
    else:
        loss_kernel = args.lambda_kernel_reg * F.mse_loss(kernel_pred, kernel_true)
        if fgsd_kernel_true is not None:
            loss_fgsd_kernel = args.lambda_fgsd_kernel_reg * F.mse_loss(fgsd_kernel_pred, fgsd_kernel_true)
        if spk_kernel_true is not None:
            loss_spk_kernel = args.lambda_spk_kernel_reg * F.mse_loss(spk_kernel_pred, spk_kernel_true)

    loss_adj_reconst = torch.FloatTensor([0]).to(device)
    # if A_pred is not None:
    #     num_nodes = A.shape[0]
    #     num_edges = torch.sparse.sum(A)
    #     edge_ratio = (num_nodes * num_nodes - num_edges) / num_edges
    #     A_true = A.to_dense() + torch.eye(num_nodes).to(device)
    #     # A_true = A.to_dense()
    #     loss_adj_reconst = args.lambda_adj_reconst_reg * F.binary_cross_entropy_with_logits(A_pred, A_true, pos_weight=edge_ratio)
    #     # loss_adj_reconst = args.lambda_adj_reconst_reg * F.mse_loss(torch.sigmoid(A_pred), A_true)

    total_loss = 0 * loss_kernel
    if args.use_class_loss:
        total_loss = total_loss + loss_class
    if args.use_wl_features:
        total_loss = total_loss + loss_kernel
    if args.use_fgsd_features:
        total_loss = total_loss + loss_fgsd_kernel
    if args.use_spk_features:
        total_loss = total_loss + loss_spk_kernel
    if args.use_adj_reconst_loss:
        total_loss = total_loss + loss_adj_reconst

    return total_loss, loss_class, loss_kernel, loss_fgsd_kernel, loss_spk_kernel, loss_adj_reconst


def train_per_epoch():
    global best_validation_loss, best_acc_validation, best_acc_validation_loss

    t = time.time()
    model.train()
    num_samples = len(train_loader.dataset)
    total_loss = 0
    total_class_loss = 0
    total_kernel_loss = 0
    total_fgsd_kernel_loss = 0
    total_spk_kernel_loss = 0
    total_adj_reconst_loss = 0
    correct = 0
    num_batches = 0
    total_batch_prep_time = 0
    X_train = []
    Y_train = []

    for idx_batch, batch in enumerate(train_loader):

        X = batch['node_feature_matrix'].to(device)
        A = batch['adjacency_matrix'].to(device)
        # A_mask = batch['adjacency_mask'].to(device)
        A_mask = None
        batch_sample_matrix = batch['batch_sample_matrix'].to(device)
        graph_labels = batch['graph_labels'].to(device)
        kernel_true = batch['wl_kernel_matrix'].to(device)
        class_kernel_true = None
        spk_kernel_true = None
        fgsd_kernel_true = None
        if args.use_adaptive_kernel_loss:
            class_kernel_true = batch['class_kernel_matrix'].to(device)
        if args.use_spk_features:
            spk_kernel_true = batch['spk_kernel_matrix'].to(device)
        if args.use_fgsd_features:
            fgsd_kernel_true = batch['fgsd_kernel_matrix'].to(device)

        optimizer.zero_grad()
        logits, kernel_pred, fgsd_kernel_pred, spk_kernel_pred, graph_emb, A_pred = model(X, A, A_mask, batch_sample_matrix)
        loss, loss_class, loss_kernel, loss_fgsd_kernel, loss_spk_kernel, loss_adj_reconst = compute_loss(logits, graph_labels, kernel_pred, kernel_true, fgsd_kernel_pred, fgsd_kernel_true, spk_kernel_pred, spk_kernel_true, class_kernel_true, A_pred, A)
        loss.backward()
        optimizer.step()
        optimizer_lr_scheduler.step()

        pred = logits.max(dim=1)[1]
        correct += pred.eq(graph_labels).sum().item()

        total_loss += loss.item()
        total_class_loss += loss_class.item()
        total_kernel_loss += loss_kernel.item()
        total_fgsd_kernel_loss += loss_fgsd_kernel.item()
        total_spk_kernel_loss += loss_spk_kernel.item()
        total_adj_reconst_loss += loss_adj_reconst.item()
        num_batches = num_batches + 1
        total_batch_prep_time = total_batch_prep_time + batch['prep_time']
        X_train.append(graph_emb.data.cpu().numpy())
        Y_train.append(graph_labels.data.cpu().numpy())

    acc_train = correct / num_samples
    loss_train_per_sample = total_loss / num_batches
    loss_train_class_per_sample = total_class_loss / num_batches
    loss_kernel_train_per_sample = total_kernel_loss / num_batches
    loss_fgsd_kernel_train_per_sample = total_fgsd_kernel_loss / num_batches
    loss_spk_kernel_train_per_sample = total_spk_kernel_loss / num_batches
    loss_adj_reconst_train_per_sample = total_adj_reconst_loss / num_batches

    # loss_train_per_sample, loss_train_class_per_sample, loss_kernel_train_per_sample, loss_fgsd_kernel_train_per_sample, loss_spk_kernel_train_per_sample, loss_adj_reconst_train_per_sample, acc_train, X_train, Y_train = eval_model(train_loader)
    loss_validation_per_sample, loss_validation_class_per_sample, loss_kernel_validation_per_sample, loss_fgsd_kernel_validation_per_sample, loss_spk_kernel_validation_per_sample, loss_adj_reconst_validation_per_sample, acc_validation, X_validation, Y_validation = eval_model(validation_loader)

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    X_validation = np.concatenate(X_validation)
    Y_validation = np.concatenate(Y_validation)

    if args.use_svm_classifier:
        clf = svm.SVC(C=1, gamma='scale', class_weight='balanced')
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_validation)
        acc_validation = accuracy_score(Y_validation, Y_pred)
        Y_pred = clf.predict(X_train)
        acc_train = accuracy_score(Y_train, Y_pred)

    if best_acc_validation < acc_validation:
        best_acc_validation = acc_validation
    if best_validation_loss > loss_validation_per_sample:
        best_validation_loss = loss_validation_per_sample
        best_acc_validation_loss = acc_validation

    writer.add_scalars('acc/best', {'best_acc_validation_loss': best_acc_validation_loss, 'best_acc_validation': best_acc_validation}, epoch + 1)
    writer.add_scalars('acc', {'acc_train': acc_train, 'acc_validation': acc_validation}, epoch + 1)
    writer.add_scalars('loss/loss_total', {'loss_train_per_sample': loss_train_per_sample, 'loss_validation_per_sample': loss_validation_per_sample}, epoch + 1)
    writer.add_scalars('loss/loss_wl', {'loss_kernel_train_per_sample': loss_kernel_train_per_sample, 'loss_kernel_validation_per_sample': loss_kernel_validation_per_sample}, epoch + 1)
    writer.add_scalars('loss/loss_fgsd', {'loss_fgsd_kernel_train_per_sample': loss_fgsd_kernel_train_per_sample, 'loss_fgsd_kernel_validation_per_sample': loss_fgsd_kernel_validation_per_sample}, epoch + 1)
    writer.add_scalars('loss/loss_spk', {'loss_spk_kernel_train_per_sample': loss_spk_kernel_train_per_sample, 'loss_spk_kernel_validation_per_sample': loss_spk_kernel_validation_per_sample}, epoch + 1)
    writer.add_scalars('loss/loss_adj', {'loss_adj_reconst_train_per_sample': loss_adj_reconst_train_per_sample, 'loss_adj_reconst_validation_per_sample': loss_adj_reconst_validation_per_sample}, epoch + 1)

    logging.info('Epoch: {:04d}'.format(epoch + 1) +
                 ' acc_train: {:.4f}'.format(acc_train) +
                 ' acc_validation: {:.4f}'.format(acc_validation) +
                 ' best_acc_validation: {:.4f}'.format(best_acc_validation) +
                 ' best_acc_validation_loss: {:.4f}'.format(best_acc_validation_loss) +
                 ' loss_train: {:08.5f}'.format(loss_train_per_sample) +
                 ' loss_class_train: {:08.5f}'.format(loss_train_class_per_sample) +
                 ' loss_kernel_train: {:08.5f}'.format(loss_kernel_train_per_sample) +
                 ' loss_fgsd_kernel_train: {:08.5f}'.format(loss_fgsd_kernel_train_per_sample) +
                 ' loss_spk_kernel_train: {:08.5f}'.format(loss_spk_kernel_train_per_sample) +
                 ' loss_adj_reconst_train: {:08.5f}'.format(loss_adj_reconst_train_per_sample) +
                 ' loss_validation: {:08.5f}'.format(loss_validation_per_sample) +
                 ' loss_class_validation: {:08.5f}'.format(loss_validation_class_per_sample) +
                 ' loss_kernel_validation: {:08.5f}'.format(loss_kernel_validation_per_sample) +
                 ' loss_fgsd_kernel_validation: {:08.5f}'.format(loss_fgsd_kernel_validation_per_sample) +
                 ' loss_spk_kernel_validation: {:08.5f}'.format(loss_spk_kernel_validation_per_sample) +
                 ' loss_adj_reconst_validation: {:08.5f}'.format(loss_adj_reconst_validation_per_sample) +
                 ' lr: {:.2e}'.format(optimizer.param_groups[0]['lr']) +
                 ' batch_prep_time: {:.4f}s'.format(total_batch_prep_time) +
                 ' crossval_split: {:04d}'.format(args.crossval_split) +
                 ' time: {:.4f}s'.format(time.time() - t))

    with experiment.train():
        experiment.log_metric("loss", loss_train_per_sample, step=epoch)
        experiment.log_metric("accuracy", float('{:.4f}'.format(acc_train)), step=epoch)
        experiment.log_metric("loss_train", float('{:.4f}'.format(loss_train_per_sample)), step=epoch)
        experiment.log_metric("loss_class_train", float('{:.4f}'.format(loss_train_class_per_sample)), step=epoch)
        experiment.log_metric("loss_kernel_train", float('{:.4f}'.format(loss_kernel_train_per_sample)), step=epoch)
        experiment.log_metric("loss_fgsd_kernel_train", float('{:.4f}'.format(loss_fgsd_kernel_train_per_sample)), step=epoch)
        experiment.log_metric("loss_spk_kernel_train", float('{:.4f}'.format(loss_spk_kernel_train_per_sample)), step=epoch)
        experiment.log_metric("loss_adj_reconst_train", float('{:.4f}'.format(loss_adj_reconst_train_per_sample)), step=epoch)

    with experiment.validation():
        experiment.log_metric("loss", loss_validation_per_sample, step=epoch)
        experiment.log_metric("accuracy", float('{:.4f}'.format(acc_validation)), step=epoch)
        experiment.log_metric("best_acc", float('{:.4f}'.format(best_acc_validation)), step=epoch)
        experiment.log_metric("loss_best_acc", float('{:.4f}'.format(best_acc_validation_loss)), step=epoch)
        experiment.log_metric("epoch", float('{:04d}'.format(epoch + 1)), step=epoch)
        experiment.log_metric("loss_validation", float('{:.4f}'.format(loss_validation_per_sample)), step=epoch)
        experiment.log_metric("loss_class_validation", float('{:.4f}'.format(loss_validation_class_per_sample)), step=epoch)
        experiment.log_metric("loss_kernel_validation", float('{:.4f}'.format(loss_kernel_validation_per_sample)), step=epoch)
        experiment.log_metric("loss_fgsd_kernel_validation", float('{:.4f}'.format(loss_fgsd_kernel_validation_per_sample)), step=epoch)
        experiment.log_metric("loss_spk_kernel_validation", float('{:.4f}'.format(loss_spk_kernel_validation_per_sample)), step=epoch)
        experiment.log_metric("loss_adj_reconst_validation", float('{:.4f}'.format(loss_adj_reconst_validation_per_sample)), step=epoch)


if __name__ == '__main__':

    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if 'pydevd' in sys.modules:
        DEBUGGING = True
    else:
        DEBUGGING = False

    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--pretrained_model_file', type=str, default=None)
    parser.add_argument('--gpu_device', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default='DD')
    parser.add_argument('--crossval_split', type=int, default=2)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--run_on_comet', type=lambda x: (str(x).lower() == 'true'), default=not DEBUGGING)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--drop_prob', type=float, default=0)
    parser.add_argument('--num_gconv_layers', type=int, default=5)
    parser.add_argument('--num_gfc_layers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=1000)

    parser.add_argument('--use_class_loss', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_wl_features', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_fgsd_features', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_spk_features', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_adj_reconst_loss', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_adaptive_kernel_loss', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_svm_classifier', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--lambda_class_reg', type=float, default=1.0)
    parser.add_argument('--lambda_kernel_reg', type=float, default=1.0)
    parser.add_argument('--lambda_fgsd_kernel_reg', type=float, default=1.0)
    parser.add_argument('--lambda_spk_kernel_reg', type=float, default=1.0)
    parser.add_argument('--lambda_adj_reconst_reg', type=float, default=1.0)

    parser.add_argument('--warmup_epochs', type=float, nargs='*', default=[2.0], help='Number of epochs during which learning rate increases linearly from init_lr to max_lr. Afterwards, learning rate decreases exponentially from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, nargs='*', default=[1e-4], help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, nargs='*', default=[1e-3], help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, nargs='*', default=[1e-4], help='Final learning rate')
    parser.add_argument('--lr_scaler', type=float, nargs='*', default=[1.0], help='Amount by which to scale init_lr, max_lr, and final_lr (for convenience)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='lr decay per epoch, for decay scheduler')

    args, unknown = parser.parse_known_args()

    experiment = Experiment(api_key=API_KEY, project_name="universal-graph-embedding", workspace="saurabh08", disabled=not args.run_on_comet)
    experiment_id = experiment.get_key()

    data_path = os.path.join(args.data_dir, args.dataset_name)
    log_path = os.path.join(args.log_dir, experiment_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(log_path, 'console_output.txt'))])

    run_filepath = os.path.abspath(__file__)
    shutil.copy(run_filepath, log_path)
    src_list = ['./train', './utils', './torch_dgl', './dataloader', './config']
    dest_list = [os.path.join(log_path, 'train'), os.path.join(log_path, 'utils'), os.path.join(log_path, 'torch_dgl'), os.path.join(log_path, 'dataloader'), os.path.join(log_path, 'config')]
    for src, dest in zip(src_list, dest_list):
            shutil.copytree(src, dest)

    for arg, value in sorted(vars(args).items()):
        logging.info("Hyperparameter: %s: %r", arg, value)

    writer = SummaryWriter('tensorboard/')

    dataset = TUDataset(data_path, name=args.dataset_name, shuffle=False, compute_graph_kernel_features=True, wl_node_labels='node_label')
    cross_val_path = os.path.join(args.data_dir, args.dataset_name, 'crossval_10fold_idx/')
    idx_train = np.loadtxt(cross_val_path + 'idx_train_split_' + str(args.crossval_split) + '.txt', dtype=np.int64)
    idx_validation = np.loadtxt(cross_val_path + 'idx_validation_split_' + str(args.crossval_split) + '.txt', dtype=np.int64)
    train_dataset = dataset[idx_train]
    validation_dataset = dataset[idx_validation]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, batch_prep_func=batch_prep_input)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, batch_prep_func=batch_prep_input)

    num_train_samples = len(train_dataset)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    device = torch.device('cuda', args.gpu_device)

    if args.use_spk_features:
        spk_path = os.path.join(args.data_dir, args.dataset_name, args.dataset_name + '_K_shorvalidation_path.npy')
        spk_kernel = np.load(spk_path)
    if args.use_fgsd_features:
        fgsd_path = os.path.join(args.data_dir, args.dataset_name, args.dataset_name + '_K_fgsd.npy')
        fgsd_kernel = np.load(fgsd_path)

    ####################################

    model = UniversalGraphEmbedder(num_features, num_classes, args.hidden_dim, args.num_gconv_layers, args.num_gfc_layers, args.drop_prob, args.use_fgsd_features, args.use_spk_features).to(device)
    loss_func = nn.MSELoss()

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # lr_decay_factor = 0.1
    # lr_decay_at_epochs = 50
    # optimizer_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_at_epochs, gamma=lr_decay_factor)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr[0], weight_decay=0.1)
    optimizer_lr_scheduler = NoamLR(optimizer=optimizer, warmup_epochs=args.warmup_epochs,
                                    total_epochs=[args.num_epochs],
                                    steps_per_epoch=num_train_samples // args.batch_size,
                                    init_lr=args.init_lr, max_lr=args.max_lr, final_lr=args.final_lr)

    logging.info('Starting epoch: {:04d}'.format(1) + ' current optimizer lr: {:.2e}'.format(optimizer.param_groups[0]['lr']))

    if args.pretrained_model_file is not None:
        logging.info('Loading previous trained model file from: ' + str(args.pretrained_model_file))
        pretrain_model_state = load_model(args.pretrained_model_file)
        model_state_dict = load_partial_model_state(model.state_dict(), pretrain_model_state['model_state_dict'])
        model.load_state_dict(model_state_dict)
        # optimizer.load_state_dict(model_state['optimizer_state_dict'])
    else:
        logging.info('Training from scratch... ')

    best_validation_loss = 1e10
    best_acc_validation = 0.0
    best_acc_validation_loss = 0.0
    for epoch in range(args.num_epochs):

        train_per_epoch()
        # optimizer_lr_scheduler.step()

        if (epoch+1) % args.save_steps == 0:
            model_checkpoint_name = "model_state_epoch_" + str(epoch+1) + ".pt"
            save_path = os.path.join(log_path, model_checkpoint_name)
            save_model(model, optimizer, save_path, device)

    writer.close()

