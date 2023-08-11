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

common_data_path = os.path.join(projects_folder, "latent_circuit_inference", "data", "inferred_LCs",
                         "0.011325_CDDM;relu;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.002;maxiter=3000",
                         "0.9563516149290711_0.9738313572216338_LC_8nodes;encoding")
circuit_data_path = os.path.join(common_data_path, "0.9563516149290711_0.9738313572216338_LC_params.json")
net_params = json.load(open(circuit_data_path, "r+"))
task_data_path = os.path.join(common_data_path, "0.011325_config.json")
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
V = Vizualizer(G, inputs, traces, outputs)
V.vizualize()