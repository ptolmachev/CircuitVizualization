import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
projects_folder = str(Path.home()) + "/Documents/GitHub/"
sys.path.append(projects_folder)
from CircuitVizualization.src.Graph import Graph
from CircuitVizualization.src.Vizializer import Vizualizer
from rnn_coach.src.RNN_numpy import RNN_numpy
from rnn_coach.src.Task import TaskCDDMtanh
import os
import json

data = pickle.load(open("../../data/blockDM_neural_traces_example.pkl", "rb+"))
inputs = data["inputs"]
traces = data["traces"]
outputs = data["outputs"]
net_params = data["net_params"]

W_inp = np.array(net_params["W_inp"])
W_rec = np.array(net_params["W_rec"])
W_out = np.array(net_params["W_out"])

inp_node_labels = ["block", "signal"]
rec_node_labels = np.arange(W_rec.shape[0]).tolist()
out_node_labels = ["choice"]
G = Graph(W_inp=W_inp, W_rec=W_rec, W_out=W_out,
          inp_node_labels=inp_node_labels,
          rec_node_labels=rec_node_labels,
          out_node_labels=out_node_labels, label_fontsize=9)
V = Vizualizer(G, inputs, traces, outputs)
V.vizualize()