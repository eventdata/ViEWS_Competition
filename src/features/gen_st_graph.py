'''Get data ready for DGL package
Author: Yi-Fan Li
Input: This file reads the prio-grid level ucdp monthly data
Output: This file returns graph of monthly data
'''

from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from data_utils import Bi_Dict, get_subgraph
import matplotlib.pyplot as plt
import torch as th
import dgl
import random
import argparse
import copy

def get_nodes_edges(relation_dir, data_dir):
    # Read prio-grid coordinates
    df_priogrid = pd.read_csv(relation_dir, delimiter=',')
    # Get the cell id range for Africa:
    df_ucdp_month = pd.read_csv(data_dir, delimiter=',')
    gid_set = set()
    for item in df_ucdp_month['pg_id']:
        if str(item) not in gid_set:
            gid_set.add(str(item))

    # Read prio-grid info and transform that into a double direction dictionary
    bi_dict = Bi_Dict() # Initialize a bi-directional dictionary
    for idx in df_priogrid.index:
        gid = str(df_priogrid['gid'][idx])
        gid_row_col = (str(df_priogrid['row'][idx]), str(df_priogrid['col'][idx]))
        if gid in gid_set:
            bi_dict[gid] = gid_row_col

    # Search for the adjuncted 8 cells by bfs, and put the gid into a list
    d = dict() # d is a dictionary of key: gid, val: list(adjuncted gid)
    for gid in gid_set:
        d[gid] = []
        c_loc = bi_dict[gid][0] # c_row, c_col are row index and col index of the target cell
        adj_cells = bfs(c_loc[0], c_loc[1])
        for cell_loc in adj_cells:
            if cell_loc in bi_dict:
                d[gid].append(bi_dict[cell_loc][0])
    edge_set = set()
    for key, vals in d.items():
        for val in vals:
            if (key, val) not in edge_set and (val, key) not in edge_set:
                edge_set.add((key, val))
    return gid_set, edge_set

def get_graph(data_dir, node_set, edge_set):

    # Read data into dataframe:
    df_ucdp_month = pd.read_csv(data_dir, delimiter=',')
    row_nums, col_nums = df_ucdp_month.shape # Get the number of rows and cols of the data
    s_idx = 0
    cur_monthid = df_ucdp_month.iloc[s_idx]['month_id'] # month_id identifies every month in the dataset

    # Initialize graph. Since the skeleton of graph for Africa is the same for different months, we
    #   can initialize once and build the graph for every month based on this skeleton
    # Now we create the graph in networkx
    G = nx.Graph()
    G.add_nodes_from(node_set) # Add nodes
    G.add_edges_from(edge_set)
    # Convert networkx graph to dgl graph
    g = dgl.DGLGraph()
    g.from_networkx(G)

    # Iterate each row to create monthly data, and slicing the dataframe by index
    for idx, row in df_ucdp_month.iterrows():
        if row['month_id'] != cur_monthid:
            cur_year, cur_month = str(row['year']), str(row['month'])
            df_tmp = df_ucdp_month[s_idx:idx]
            # Create the graph data for this month.
            # Filtering pid only in node_set
            is_Africa = df_tmp['pg_id'].isin(node_set)
            df_tmp = df_tmp[is_Africa]

            # features = ['ged_dummy_sb', 'ged_count_sb', 'ged_best_sb', 'ged_dummy_ns', 
            #             'ged_count_ns', 'ged_best_ns', 'ged_dummy_os', 'ged_count_os', 
            #             'ged_best_os'] # Is the name necessary?

            g_tmp = copy.deepcopy(g)
            
            # Iterate through the df_tmp and assign node values to each node
            for idx, row in df_tmp.iterrows():
                val = row.values[6:]
                print(row)
            year_idx = df_tmp.iloc[0]['year']
            month_idx = df_tmp.iloc[0]['month']

            print(df_tmp)

            s_idx = idx
    return G

def bfs(row, col):
    dirs = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    res = []
    for d_row, d_col in dirs:
        res.append((str(int(row) + d_row), str(int(col) + d_col)))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get_Graph')
    # parser.set_defaults()
    
    # Set dir to files containing relations and data
    relation_dir = './data/priogrid_relations.csv'
    data_dir = './data/ucdp_month_priogrid.csv'

    node_set, edge_set = get_nodes_edges(relation_dir, data_dir)
    g = get_graph(data_dir, node_set, edge_set)


    # nx.draw(g_nx, node_size=5)
    sg_nx = get_subgraph(g_nx, sample_size=1000)
    nx.draw(sg_nx, node_size=5)
    # g_dgl = dgl.DGLGraph(g_nx)
    plt.show()

