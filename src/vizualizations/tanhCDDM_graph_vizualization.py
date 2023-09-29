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
from rnn_coach.src.Task import TaskCDDM
import os
import json

# data = pickle.load(open("blockDM_neural_traces_example.pkl", "rb+"))
# inputs = data["inputs"]
# traces = data["traces"]
# outputs = data["outputs"]
# net_params = data["net_params"]
activation = 'tanh'
RNN_score = '0.0116744'

path = os.path.join(projects_folder, "latent_circuit_inference", "data", "inferred_LCs", f"CDDM{activation}",
                         f"{RNN_score}_CDDM;{activation};N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500")
subfolders = os.listdir(path)
r_scores_full = []
r_scores_proj = []
for sf in subfolders:
    if sf == '.DS_Store':
        pass
    else:
        r_scores_full.append(float(sf.split("_")[0]))
        r_scores_proj.append(float(sf.split("_")[1]))
ind = r_scores_full.index(max(r_scores_full))
r_score_full = r_scores_full[ind]
r_score_proj = r_scores_proj[ind]
print(r_score_full)
print(r_score_proj)

common_data_path = os.path.join(path, f"{r_score_full}_{r_score_proj}_LC_CDDM{activation}")

circuit_data_path = os.path.join(common_data_path, f"{r_score_full}_{r_score_proj}_LC_params.json")
# circuit_data_path = os.path.join(projects_folder, "PRuNNe", "data", "0.0117232_CDDM;relu;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000", "pruning", "params_33nrns_mse=0.023.pkl")
net_params = json.load(open(circuit_data_path, "r+"))
# net_params = pickle.load(open(circuit_data_path, "rb+"))
task_data_path = os.path.join(common_data_path, f"{RNN_score}_config.json")
task_data = json.load(open(task_data_path, "r+"))
task_params = task_data["task_params"]
n_steps = task_data["n_steps"]
n_inputs = task_params["n_inputs"]
n_outputs = task_params["n_outputs"]

task = TaskCDDM(n_steps=n_steps, n_inputs=n_inputs, n_outputs=n_outputs, task_params=task_params)
inputs, targets, conditions = task.get_batch()
match task_data["activation"]:
    case 'relu': activation_fun = lambda x: np.maximum(0, x)
    case 'tanh': activation_fun = lambda x: np.tanh(x)

W_inp = np.array(net_params["W_inp"])
W_rec = np.array(net_params["W_rec"])
W_out = np.array(net_params["W_out"])
bias_rec = np.zeros(W_rec.shape[0])
y_init = np.zeros(W_rec.shape[0])

circuit = RNN_numpy(N=net_params["N"],
                    dt=net_params["dt"],
                      tau=net_params["tau"],
                      activation=activation_fun,
                      W_inp=np.array(net_params["W_inp"]),
                      W_rec=np.array(net_params["W_rec"]),
                      W_out=np.array(net_params["W_out"]),
                      bias_rec=bias_rec,
                      y_init=y_init)
circuit.clear_history()
circuit.run(inputs)
traces = circuit.get_history()
outputs = circuit.get_output()

inp_node_labels = ["ctx M", "ctx C", "mR", "mL", "cR", "cL"]
rec_node_labels = ["ctx M", "ctx C", "mR", "mL", "cR", "cL", "OutR", "OutL"]
out_node_labels = ["OutR", "OutL"]
G = Graph(W_inp=W_inp, W_rec=W_rec, W_out=W_out,
          inp_node_labels=inp_node_labels,
          rec_node_labels=rec_node_labels,
          out_node_labels=out_node_labels, label_fontsize=9)
# G = Graph(W_inp=W_inp, W_rec=W_rec, W_out=W_out, label_fontsize=9, rec_r = 0.03)
V = Vizualizer(G, inputs, traces, outputs)
V.vizualize()