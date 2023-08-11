import os
import numpy as np
from Graph import *
import pickle
import json
from pathlib import Path
projects_folder = str(Path.home()) + "/Documents/GitHub/"
sys.path.append(projects_folder)

path_to_data = os.path.join(projects_folder, "latent_circuit_inference", "../data", "inferred_LCs", "BlockDMtanh",
                            "0.0011968_BlockDMtanh;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000",
                            "9_0.9810267019249379_0.982110291149382_LC_BlockDMtanh",
                            "0.9810267019249379_0.982110291149382_LC_params.json")
data = json.load(open(path_to_data, "rb+"))
W_inp = np.array(data["W_inp"])
W_rec = np.array(data["W_rec"])
W_out = np.array(data["W_out"])
inp_node_labels = ["block", "signal"]
out_node_labels = ["choice"]
G = Graph(W_inp=W_inp, W_rec=W_rec, W_out = W_out,
          inp_node_labels=inp_node_labels,
          out_node_labels=out_node_labels, label_fontsize=12)
G.draw_graph()
plt.show()