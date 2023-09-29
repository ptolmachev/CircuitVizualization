from matplotlib.widgets import Slider
import numpy as np
from matplotlib import pyplot as plt

class Vizualizer():
    def __init__(self, Graph, inputs, traces, outputs):
        self.G = Graph
        self.inputs = inputs
        self.traces = traces
        self.outputs = outputs
        fig, ax = self.G.draw_graph()
        self.axt = fig.add_axes([0.25, 0.1, 0.55, 0.03])

        self.t_slider = Slider(
            ax=self.axt,
            label='time, ms',
            valmin=0.0,
            valmax=inputs.shape[1],
            valinit=0,
            valstep=np.arange(inputs.shape[1]))

        axtrial = fig.add_axes([0.1, 0.25, 0.0225, 0.53])
        self.trial_slider = Slider(
            ax=axtrial,
            label="trial number",
            valmin=0,
            valmax=inputs.shape[-1],
            valinit=0,
            valstep=np.arange(inputs.shape[-1]),
            orientation="vertical"
        )

    def update(self, val):
        t = int(self.t_slider.val)
        trial_num = int(self.trial_slider.val)
        # update values of a graph
        u = self.inputs[:, t, trial_num]
        a = self.traces[:, t, trial_num]
        o = self.outputs[:, t, trial_num]
        node_labels = self.G.label_dict["rec"]
        inp_labels = self.G.label_dict["inp"]
        out_labels = self.G.label_dict["out"]
        A = np.max([np.abs(np.min(self.traces)), np.abs(np.max(self.traces))])
        U = np.max([np.abs(np.min(self.inputs)), np.abs(np.max(self.inputs))])
        O = np.max([np.abs(np.min(self.outputs)), np.abs(np.max(self.outputs))])

        # change the colors of nodes
        for i in range(len(node_labels)):
            if a[i] >= 0:
                self.G.node_dict["rec"][node_labels[i]]["facecolor"] = (1, 0, 0, np.abs(a[i]) / A)
            else:
                self.G.node_dict["rec"][node_labels[i]]["facecolor"] = (0, 0, 1, np.abs(a[i]) / A)
        for i in range(len(inp_labels)):
            if u[i] >= 0:
                self.G.node_dict["inp"][inp_labels[i]]["facecolor"] = (1, 0, 0, np.abs(u[i]) / U)
            else:
                self.G.node_dict["inp"][inp_labels[i]]["facecolor"] = (0, 0, 1, np.abs(u[i]) / U)
        for i in range(len(out_labels)):
            if o[i] >= 0:
                self.G.node_dict["out"][out_labels[i]]["facecolor"] = (1, 0, 0, np.abs(o[i]) / O)
            else:
                self.G.node_dict["out"][out_labels[i]]["facecolor"] = (0, 0, 1, np.abs(o[i]) / O)
        # change the strength of connections:
        # for each connection in the edge dicts, modify the strength by multiplying it by a[i]
        max_W = np.max(np.abs(self.G.W_rec))
        for edge in (self.G.edge_dict["rec"]).keys():
            node_from = edge[0]
            node_to = edge[1]
            strength = self.G.edge_dict["rec"][(node_from, node_to)]["strength"]
            new_strength = np.abs(a[node_from]) * np.sign(self.G.W_rec[node_to, node_from]) * (
                        self.G.W_rec[node_to, node_from] / max_W) ** 2
            self.G.edge_dict["rec"][(node_from, node_to)]["strength"] = new_strength
            color = self.G.default_pos_clr if (a[node_from] * self.G.W_rec[node_to, node_from] >= 0) else self.G.default_neg_clr
            self.G.edge_dict["rec"][(node_from, node_to)]["color"] = color
            self.G.edge_dict["rec"][(node_from, node_to)]["capstyle"] = 'arrow' if (a[node_from] * self.G.W_rec[node_to, node_from] >= 0) else 'circle'
            self.G.edge_dict["rec"][(node_from, node_to)]["lw"] = 2 * strength

        max_W = np.max(np.abs(self.G.W_inp))
        for edge in (self.G.edge_dict["inp"]).keys():
            node_from = edge[0]
            node_to = edge[1]
            strength = self.G.edge_dict["inp"][(node_from, node_to)]["strength"]
            new_strength = np.abs(u[node_from]) * np.sign(self.G.W_inp[node_to, node_from]) * (
                        self.G.W_inp[node_to, node_from] / max_W) ** 2
            self.G.edge_dict["inp"][(node_from, node_to)]["strength"] = new_strength
            color = self.G.default_pos_clr if (u[node_from] >= 0) else self.G.default_neg_clr
            self.G.edge_dict["inp"][(node_from, node_to)]["color"] = color
            self.G.edge_dict["inp"][(node_from, node_to)]["capstyle"] = 'arrow' if (u[node_from] >= 0) else 'circle'
            self.G.edge_dict["inp"][(node_from, node_to)]["lw"] = 2 * strength

        max_W = np.max(np.abs(self.G.W_out))
        for edge in (self.G.edge_dict["out"]).keys():
            node_from = edge[0]
            node_to = edge[1]
            strength = self.G.edge_dict["out"][(node_from, node_to)]["strength"]
            new_strength = np.abs(a[node_from]) * np.sign(self.G.W_out[node_to, node_from]) * (
                        self.G.W_out[node_to, node_from] / max_W) ** 2
            self.G.edge_dict["out"][(node_from, node_to)]["strength"] = new_strength
            color = self.G.default_pos_clr if (a[node_from] >= 0) else self.G.default_neg_clr
            self.G.edge_dict["out"][(node_from, node_to)]["color"] = color
            self.G.edge_dict["out"][(node_from, node_to)]["capstyle"] = 'arrow' if (a[node_from] >= 0) else 'circle'
            self.G.edge_dict["out"][(node_from, node_to)]["lw"] = 2 * strength

        self.G.clear()
        self.G.draw_nodes()
        self.G.draw_edges()

        self.G.painter.figure.canvas.draw_idle()

    def vizualize(self):
        self.t_slider.on_changed(self.update)
        self.trial_slider.on_changed(self.update)
        plt.show()
