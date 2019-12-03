import numpy as np
import pandas as pd


def read_from_sdf(fname):

    file = open(fname, 'r')

    graph_list = []
    while True:
        G = _read_sdf_molecule(file)
        if G == False:
            break
        graph_list.append(G)

    file.close()
    return graph_list


# read a single molecule from file
def _read_sdf_molecule(file):
    # read the header 3 lines
    line = file.readline().split('\t')
    if line[0] == '':
        return False
    gdb_id = int(line[0].split(' ')[1])
    for i in range(2):
        file.readline()
    line = file.readline()

    # this does not work for 123456 which must be 123 and 456
    # (atoms, bonds) = [t(s) for t,s in zip((int,int),line.split())]
    num_atoms = int(line[:3])
    num_bonds = int(line[3:6])

    v = []
    node_labels = []
    for i in range(num_atoms):
        line = file.readline()
        atom_symbol = line.split()[3]
        v.append(i + 1)
        node_labels.append(atom_symbol)

    edge_list = []
    for i in range(num_bonds):
        line = file.readline()
        u = int(line[:3]) - 1
        v = int(line[3:6]) - 1
        edge_list.append((u, v))

    while line != '':
        line = file.readline()
        if line[:4] == "$$$$":
            break

    G = dict()
    G['node_labels'] = node_labels
    G['edge_list'] = edge_list
    G['gdb_id'] = gdb_id
    G['num_nodes'] = num_atoms
    G['num_edges'] = num_bonds

    return G
