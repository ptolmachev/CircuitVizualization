import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from CircuitVizualization.src.utils import Rotate
from scipy.optimize import linear_sum_assignment

class Painter():
    '''
    An internal class of Graph: contains figure and axes, as well as the patches which needs to be drawn on the figure
    has an internal method of drawing a directed curved arrow
    '''
    def __init__(self, figsize=(10, 10)):
        fig_circuit, ax = plt.subplots(1, 1, figsize=figsize)
        self.figure = fig_circuit
        self.ax = ax
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_aspect('equal')
        self.ax.axis('off')

    def Bezier_arrow(self, posA, posB,
                     capstyle,
                     bezier_point=None,
                     color='r',
                     alpha=1.0,
                     lw=2,
                     theta=np.pi / 8,
                     head_shaft_ratio=0.05):

        Path = mpath.Path
        if bezier_point is None:
            bezier_point = (np.array(posB) + np.array(posA)) / 2

        path_data = [(Path.MOVETO, posA),
                     (Path.CURVE3, bezier_point),
                     (Path.LINETO, posB)]
        codes, verts = zip(*path_data)
        path_shaft = mpath.Path(verts, codes)
        patch_shaft = mpatches.PathPatch(path_shaft, fill=False, alpha=alpha, edgecolor=color, lw=lw, capstyle='round')

        central_shaft = -(np.array(posB) - np.array(bezier_point))
        central_shaft /= np.linalg.norm(central_shaft)
        path_data = []
        if capstyle == 'arrow':
            init_point = np.array(posB) + 0.5 * head_shaft_ratio * central_shaft
            left_arrow_point = head_shaft_ratio * (central_shaft @ Rotate(theta)) + np.array(posB)
            right_arrow_point = head_shaft_ratio * (central_shaft @ Rotate(-theta)) + np.array(posB)
            path_data.extend([(Path.MOVETO, init_point),
                              (Path.LINETO, left_arrow_point),
                              (Path.LINETO, posB),
                              (Path.LINETO, right_arrow_point),
                              (Path.CLOSEPOLY, (0, 0))])
            codes, verts = zip(*path_data)
            path_arrow = mpath.Path(verts, codes)
            patch_arrow = mpatches.PathPatch(path_arrow, fill=True, alpha=alpha/2, edgecolor=color, facecolor=color,
                                             lw=lw)
        if capstyle == 'circle':
            N = 24
            point = (np.array(posB) - 0.25 * head_shaft_ratio * central_shaft)
            path_data.append((Path.MOVETO, point))
            for n in range(N):
                point = np.array(posB) - 0.25 * head_shaft_ratio * central_shaft @ Rotate((n / N) * 2 * np.pi)
                path_data.append((Path.LINETO, point))
            path_data.append((Path.CLOSEPOLY, point))
            codes, verts = zip(*path_data)
            path_arrow = mpath.Path(verts, codes)
            patch_arrow = mpatches.PathPatch(path_arrow, fill=True, alpha=alpha/2, edgecolor='k', facecolor=color, lw=1)
        return patch_shaft, patch_arrow


class Graph():
    def __init__(self, W_inp, W_rec, W_out,
                 inp_node_labels=None,
                 inp_node_positions=None,
                 rec_node_labels=None,
                 rec_node_positions=None,
                 out_node_labels=None,
                 out_node_positions=None,
                 R = 0.6,
                 cutoff_weight=0.05,
                 rec_r = 0.07,
                 out_r = 0.03,
                 inp_r = 0.03,
                 inp_line_offset = 0.2,
                 out_line_offset = 0.2,
                 inp_node_distance=0.2,
                 out_node_distance=0.2,
                 default_pos_clr='r',
                 default_neg_clr='b',
                 label_fontsize = 9,
                 figsize = (7, 7)):
        self.W_inp = W_inp
        self.W_rec = W_rec
        self.W_out = W_out
        self.n_rec = W_rec.shape[0]
        self.n_inp = W_inp.shape[1] if not (W_inp is None) else 0
        self.n_out = W_out.shape[0] if not (W_out is None) else 0
        self.R = R
        self.rec_r = rec_r
        self.inp_r = inp_r
        self.out_r = out_r
        self.inp_line_offset = inp_line_offset
        self.out_line_offset = out_line_offset
        self.inp_node_distance = inp_node_distance
        self.out_node_distance = out_node_distance
        self.cutoff_weight = cutoff_weight
        self.label_fontsize = label_fontsize

        self.edge_dict = {}
        self.label_dict = {}
        self.node_dict = {}
        self.position_dict = {}
        for key in ["inp", "out", "rec"]: #make sure "rec" is last cause rec layout depends on inp and out
            self.edge_dict[key] = {}
            self.label_dict[key] = {}
            self.node_dict[key] = {}
            self.position_dict[key] = {}

            if not (eval(f"{key}_node_labels") is None):
                self.label_dict[key] = eval(f"{key}_node_labels")
            else:
                self.label_dict[key] = np.arange(eval(f"self.n_{key}")).tolist()
            if (eval(f"{key}_node_positions") is None):
                self.position_dict[key] = eval(f"self.get_{key}_layout()")
            else:
                self.position_dict[key] = eval(f"{key}_node_positions")
        self.cutoff_weight = cutoff_weight
        self.default_pos_clr = default_pos_clr
        self.default_neg_clr = default_neg_clr
        self.painter = Painter(figsize=figsize)


    def get_inp_layout(self):
        x = - self.R - self.rec_r - self.inp_line_offset # the vertical line of input nodes
        start_y = (int(self.n_inp / 2)) * self.inp_node_distance
        if self.n_inp % 2 == 0:
            start_y -= self.inp_node_distance/2
        positions = [np.array([x, start_y - i * self.inp_node_distance]) for i in range(self.n_inp)]
        return positions

    def get_out_layout(self):
        x = self.R + self.rec_r + self.out_line_offset
        start_y = int(self.n_out // 2) * self.out_node_distance
        if self.n_out % 2 == 0:
            start_y -= self.out_node_distance/2
        positions = [np.array([x, start_y - i * self.out_node_distance]) for i in range(self.n_out)]
        return positions

    def get_rec_layout(self):
        phi = np.pi / self.n_rec
        positions = [self.R * np.array([np.cos(2 * np.pi * i/self.n_rec + phi), np.sin(2 * np.pi * i/self.n_rec + phi)])
                     for i in range(self.n_rec)]
        # compiling cost matrix
        C = np.zeros((self.n_rec, self.n_rec))
        for i in range(self.n_rec): #pos
            for j in range(self.n_rec): #neuron
                inp_inds = np.where(np.abs(self.W_inp[j, :]) > 0)[0]
                out_inds = np.where(np.abs(self.W_out[:, j]) > 0)[0]
                for k in inp_inds:
                    C[j, i] += np.abs(self.W_inp[j, k]) * np.sum((self.position_dict["inp"][k] - positions[i]) ** 2)
                for k in out_inds:
                    C[j, i] += np.abs(self.W_out[k, j]) * np.sum((self.position_dict["out"][k] - positions[i]) ** 2)
        row_ind, col_ind = linear_sum_assignment(C)
        new_positions = [positions[col_ind[i]] for i in range(len(col_ind))]
        return new_positions

    def set_nodes(self,
                  rec_fill=True, rec_facecolor='lightgray', rec_edgecolor='k', rec_show_label=True,
                  inp_fill=True, inp_facecolor='lightgray', inp_edgecolor='k', inp_show_label=False,
                  out_fill=True, out_facecolor='lightgray', out_edgecolor='k', out_show_label=False):
        for key in ["inp", "rec", "out"]:
            for l, label in enumerate(self.label_dict[key]):
                self.node_dict[key][label] = {}
                self.node_dict[key][label]["radius"] = eval(f"self.{key}_r")
                self.node_dict[key][label]["fill"] = eval(f"{key}_fill")
                self.node_dict[key][label]["facecolor"] = eval(f"{key}_facecolor")
                self.node_dict[key][label]["edgecolor"] = eval(f"{key}_edgecolor")
                self.node_dict[key][label]["show_label"] = eval(f"{key}_show_label")
                self.node_dict[key][label]["pos"] = self.position_dict[key][l]
        return None

    def set_rec_edge(self,
                     strength,
                     node_from,
                     node_to,
                     bezier_point,
                     color=None,
                     cap='default',
                     ls='-',
                     alpha=1.0):
        # setting by the indices, not labels!
        j = self.label_dict["rec"].index(node_from)
        i = self.label_dict["rec"].index(node_to)
        self.edge_dict["rec"][(j, i)] = {}
        self.edge_dict["rec"][(j, i)]["node_from"] = node_from
        self.edge_dict["rec"][(j, i)]["node_to"] = node_to
        self.edge_dict["rec"][(j, i)]["strength"] = strength
        if color is None:
            color = self.default_pos_clr if (strength >= 0) else self.default_neg_clr
            self.edge_dict["rec"][(j, i)]["color"] = color
        if cap == 'default':
            self.edge_dict["rec"][(j, i)]["capstyle"] = 'arrow' if (strength >= 0) else 'circle'
        self.edge_dict["rec"][(j, i)]["lw"] = 2 * strength
        self.edge_dict["rec"][(j, i)]["alpha"] = alpha
        self.edge_dict["rec"][(j, i)]["ls"] = ls
        self.edge_dict["rec"][(j, i)]["bezier_point"] = bezier_point
        return None

    def set_inp_edge(self,
                     strength,
                     inp_from,
                     node_to,
                     bezier_point,
                     color=None,
                     cap='default',
                     ls='-',
                     alpha=1.0):
        j = self.label_dict["inp"].index(inp_from)
        i = self.label_dict["rec"].index(node_to)
        self.edge_dict["inp"][(j, i)] = {}
        self.edge_dict["inp"][(j, i)]["input_from"] = inp_from
        self.edge_dict["inp"][(j, i)]["node_to"] = node_to
        self.edge_dict["inp"][(j, i)]["strength"] = strength
        if color is None:
            color = self.default_pos_clr if (strength >= 0) else self.default_neg_clr
            self.edge_dict["inp"][(j, i)]["color"] = color
        if cap == 'default':
            self.edge_dict["inp"][(j, i)]["capstyle"] = 'arrow' if (strength >= 0) else 'circle'
        self.edge_dict["inp"][(j, i)]["lw"] = 2 * strength
        self.edge_dict["inp"][(j, i)]["alpha"] = alpha
        self.edge_dict["inp"][(j, i)]["ls"] = ls
        self.edge_dict["inp"][(j, i)]["bezier_point"] = bezier_point
        return None

    def set_out_edge(self,
                     strength,
                     node_from,
                     output_to,
                     bezier_point,
                     color=None,
                     cap='default',
                     ls='-',
                     alpha=1.0):
        j = self.label_dict["rec"].index(node_from)
        i = self.label_dict["out"].index(output_to)
        self.edge_dict["out"][(j, i)] = {}
        self.edge_dict["out"][(j, i)]["node_from"] = node_from
        self.edge_dict["out"][(j, i)]["output_to"] = output_to
        self.edge_dict["out"][(j, i)]["strength"] = strength
        if color is None:
            color = self.default_pos_clr if (strength >= 0) else self.default_neg_clr
            self.edge_dict["out"][(j, i)]["color"] = color
        if cap == 'default':
            self.edge_dict["out"][(j, i)]["capstyle"] = 'arrow' if (strength >= 0) else 'circle'
        self.edge_dict["out"][(j, i)]["lw"] = 2 * strength
        self.edge_dict["out"][(j, i)]["alpha"] = alpha
        self.edge_dict["out"][(j, i)]["ls"] = ls
        self.edge_dict["out"][(j, i)]["bezier_point"] = bezier_point
        return None

    def set_rec_edges_from_matrix(self):
        max_W = np.max(np.abs(self.W_rec))
        for i in range(self.W_rec.shape[0]):
            for j in range(self.W_rec.shape[1]):
                if np.abs(self.W_rec[i, j]) >= self.cutoff_weight:
                    alpha = np.abs(self.W_rec[i, j]) / max_W
                    node_from = self.label_dict["rec"][j]
                    node_to = self.label_dict["rec"][i]
                    posA = self.node_dict["rec"][node_from]["pos"]
                    posB = self.node_dict["rec"][node_to]["pos"]
                    bezier_point = (np.array(posB) + np.array(posA)) / 2
                    strength = np.sign(self.W_rec[i, j]) * (self.W_rec[i, j]/max_W) ** 2
                    self.set_rec_edge(strength=strength,
                                      bezier_point=bezier_point,
                                      node_from=self.label_dict["rec"][j],
                                      node_to=self.label_dict["rec"][i],
                                      alpha=alpha)
        return None

    def set_inp_edges_from_matrix(self):
        max_W = np.max(np.abs(self.W_inp))
        for i in range(self.W_inp.shape[0]):
            for j in range(self.W_inp.shape[1]):
                if np.abs(self.W_inp[i, j]) >= self.cutoff_weight:
                    alpha = np.abs(self.W_inp[i, j]) / max_W
                    node_to = self.label_dict["rec"][i]
                    inp_from = self.label_dict["inp"][j]
                    posA = self.node_dict["inp"][inp_from]["pos"]
                    posB = self.node_dict["rec"][node_to]["pos"]
                    bezier_point = (np.array(posB) + np.array(posA)) / 2
                    strength = np.sign(self.W_inp[i, j]) * (self.W_inp[i, j]/max_W) ** 2
                    self.set_inp_edge(strength= strength,
                                      bezier_point=bezier_point,
                                      inp_from=self.label_dict["inp"][j],
                                      node_to=self.label_dict["rec"][i],
                                      alpha=alpha)
        return None

    def set_out_edges_from_matrix(self):
        max_W = np.max(np.abs(self.W_out))
        for i in range(self.W_out.shape[0]):
            for j in range(self.W_out.shape[1]):
                if np.abs(self.W_out[i, j]) >= self.cutoff_weight:
                    alpha = np.abs(self.W_out[i, j]) / max_W
                    node_from = self.label_dict["rec"][j]
                    out_to = self.label_dict["out"][i]
                    posA = self.node_dict["rec"][node_from]["pos"]
                    posB = self.node_dict["out"][out_to]["pos"]
                    bezier_point = (np.array(posB) + np.array(posA)) / 2
                    strength = np.sign(self.W_out[i, j]) * (self.W_out[i, j]/max_W) ** 2
                    self.set_out_edge(strength=strength,
                                      bezier_point=bezier_point,
                                      node_from=self.label_dict["rec"][j],
                                      output_to=self.label_dict["out"][i],
                                      alpha=alpha)
        return None

    def set_edges_from_matrices(self):
        self.set_inp_edges_from_matrix()
        self.set_rec_edges_from_matrix()
        self.set_out_edges_from_matrix()

    def draw_nodes(self):
        '''for all edges in the dictionary of nodes, the painter adds a corresponding patches to the figure'''
        'to run this function, the layout have to be specified first'
        for key in ["inp", "rec", "out"]:
            if not (eval(f"self.W_{key}") is None):
                for l, label in enumerate(self.label_dict[key]):
                    radius = self.node_dict[key][label]["radius"]
                    fill = self.node_dict[key][label]["fill"]
                    facecolor = self.node_dict[key][label]["facecolor"]
                    edgecolor = self.node_dict[key][label]["edgecolor"]
                    patch = mpatches.Circle(self.position_dict[key][l],
                                            radius=radius, fill=fill,
                                            facecolor=facecolor,
                                            edgecolor=edgecolor,
                                            label=label)
                    # offset for input and output nodes to annotate them nicely
                    match key:
                        case "inp": margin =  np.array([-1.6 * radius, 2.5 * radius])
                        case "out": margin = np.array([1.6 * radius, 2.5 * radius])
                        case "rec": margin = np.zeros(2)
                    lbl = self.painter.ax.annotate(label,
                                                   xy=self.position_dict[key][l] + margin,
                                                   fontsize=self.label_fontsize,
                                                   verticalalignment="center",
                                                   horizontalalignment="center")
                    self.painter.ax.add_patch(patch)
        return None

    def curve_recurrent_connections(self):
        '''
        instead of straight recurrent connections, it makes them curved by adding a bezier point close to a middle
        such that the connection arches on the right side of the dicrected connection
        '''
        for e, edge in enumerate(self.edge_dict["rec"].keys()):
            node_from = self.label_dict["rec"].index(self.edge_dict["rec"][edge]["node_from"])
            node_to = self.label_dict["rec"].index(self.edge_dict["rec"][edge]["node_to"])
            midpoint = (self.position_dict["rec"][node_to] + self.position_dict["rec"][node_from])/2
            vector = (midpoint - self.position_dict["rec"][node_from])
            # the offset vector is perpendicular to the direction of connection
            offset_vector = vector @ Rotate(np.pi/2) / np.linalg.norm(vector)
            self.edge_dict["rec"][edge]["bezier_point"] = midpoint + self.R * (1 - np.cos(np.pi / self.n_rec)) * offset_vector

    def draw_edges(self):
        '''for all edges in the dictionary of edges, the painter adds a corresponding patches to the figure'''
        for edge in list(self.edge_dict["rec"].keys()):
            node_from = self.label_dict["rec"][edge[0]]
            node_to = self.label_dict["rec"][edge[1]]
            pos_node_from = self.node_dict["rec"][node_from]["pos"]
            pos_node_to = self.node_dict["rec"][node_to]["pos"]
            bezier_point = self.edge_dict["rec"][edge]["bezier_point"]
            color = self.edge_dict["rec"][edge]["color"]
            rad_from = self.node_dict["rec"][node_from]["radius"]
            rad_to = self.node_dict["rec"][node_from]["radius"]
            direction1 = (bezier_point - pos_node_from) / np.linalg.norm(bezier_point - pos_node_from)
            direction2 = (pos_node_to - bezier_point) / np.linalg.norm(pos_node_to - bezier_point)
            posA = pos_node_from + rad_from * direction1
            posB = pos_node_to - rad_to * direction2
            patch_shaft, patch_arrow = self.painter.Bezier_arrow(posA=posA, posB=posB,
                                                                 bezier_point=bezier_point,
                                                                 color=color,
                                                                 lw=self.edge_dict["rec"][edge]["lw"],
                                                                 capstyle=self.edge_dict["rec"][edge]["capstyle"],
                                                                 alpha=self.edge_dict["rec"][edge]["alpha"])
            self.painter.ax.add_patch(patch_shaft)
            self.painter.ax.add_patch(patch_arrow)

        if not (self.W_inp is None):
            for edge in list(self.edge_dict["inp"].keys()):
                inp_from = self.label_dict["inp"][edge[0]]
                node_to = self.label_dict["rec"][edge[1]]
                pos_node_to = self.node_dict["rec"][node_to]["pos"]
                pos_inp_from = self.node_dict["inp"][inp_from]["pos"]
                color = self.edge_dict["inp"][edge]["color"]
                bezier_point = self.edge_dict["inp"][edge]["bezier_point"]
                rad_from = self.node_dict["inp"][inp_from]["radius"]
                rad_to = self.node_dict["rec"][node_to]["radius"]
                direction1 = (bezier_point - pos_inp_from) / np.linalg.norm(bezier_point - pos_inp_from)
                direction2 = (pos_node_to - bezier_point) / np.linalg.norm(pos_node_to - bezier_point)
                posA = pos_inp_from + rad_from * direction1
                posB = pos_node_to - rad_to * direction2
                patch_shaft, patch_arrow = self.painter.Bezier_arrow(posA=posA, posB=posB,
                                                                     bezier_point=bezier_point,
                                                                     color=color,
                                                                     lw=self.edge_dict["inp"][edge]["lw"],
                                                                     capstyle=self.edge_dict["inp"][edge]["capstyle"],
                                                                     alpha=self.edge_dict["inp"][edge]["alpha"])
                self.painter.ax.add_patch(patch_shaft)
                self.painter.ax.add_patch(patch_arrow)

        if not (self.W_out is None):
            for edge in list(self.edge_dict["out"].keys()):
                node_from = self.label_dict["rec"][edge[0]]
                out_to = self.label_dict["out"][edge[1]]
                pos_out_to = self.node_dict["out"][out_to]["pos"]
                pos_node_from = self.node_dict["rec"][node_from]["pos"]
                color = self.edge_dict["out"][edge]["color"]
                bezier_point = self.edge_dict["out"][edge]["bezier_point"]
                rad_from = self.node_dict["rec"][node_from]["radius"]
                rad_to = self.node_dict["out"][out_to]["radius"]
                direction1 = (bezier_point - pos_node_from) / np.linalg.norm(bezier_point - pos_node_from)
                direction2 = (pos_out_to - bezier_point) / np.linalg.norm(pos_out_to - bezier_point)
                posA = pos_node_from + rad_from * direction1
                posB = pos_out_to - rad_to * direction2

                patch_shaft, patch_arrow = self.painter.Bezier_arrow(posA=posA, posB=posB,
                                                                     bezier_point=bezier_point,
                                                                     color=color,
                                                                     lw=self.edge_dict["out"][edge]["lw"],
                                                                     capstyle=self.edge_dict["out"][edge]["capstyle"],
                                                                     alpha=self.edge_dict["out"][edge]["alpha"])
                self.painter.ax.add_patch(patch_shaft)
                self.painter.ax.add_patch(patch_arrow)
        return None

    def draw_graph(self):
        self.set_nodes()
        self.set_edges_from_matrices()
        self.curve_recurrent_connections()
        self.draw_nodes()
        self.draw_edges()
        return self.painter.figure, self.painter.ax

    def clear(self):
        # del self.painter
        # self.painter = Painter()
        # self.painter.ax.spines.clear()
        self.painter.ax.texts.clear()
        self.painter.ax.patches.clear()
        # self.painter.ax.clear()

    # def optimize_connections(self):
    #     bezier_midpoints = np.array([self.rec_edge_dict[edge]["bezier_point"] for edge in self.rec_edge_dict.keys()])
    #     bezier_midpoints += 0.1 * np.random.randn(*bezier_midpoints.shape)
    #     positions_from = np.array(
    #         [self.node_dict[self.rec_edge_dict[edge]["node_from"]]["pos"] for edge in self.rec_edge_dict.keys()])
    #     positions_to = np.array(
    #         [self.node_dict[self.rec_edge_dict[edge]["node_to"]]["pos"] for edge in self.rec_edge_dict.keys()])
    #     true_midpoints = (positions_to + positions_from) / 2
    #     node_positions = np.array(self.positions)
    #
    #     def gaussian(point1, point2, sigma=0.15):
    #         C = (1.0 / np.sqrt(2 * np.pi * sigma ** 2))
    #         return C * np.sum(np.exp(-(point1 - point2) ** 2 / (2 * sigma ** 2)))
    #
    #     def objective(x, rec_edge_dict, node_dict):
    #         n_points = true_midpoints.shape[0]
    #         n_nodes = len(node_dict.keys())
    #         # node_positions = np.array([node["pos"] for node in node_dict.keys()])
    #         bezier_midpoints = x.reshape(*true_midpoints.shape)
    #
    #         term_1 = 0
    #         term_2 = 0
    #         for e, edge in enumerate(rec_edge_dict.keys()):
    #             node_from = self.labels[edge[0]]
    #             node_to = self.labels[edge[1]]
    #             midpoint = (node_dict[node_to]["pos"] + node_dict[node_from]["pos"]) / 2
    #             dist = (node_dict[node_to]["pos"] - node_dict[node_from]["pos"])
    #             direction = dist / np.linalg.norm(dist)
    #             right_dir = direction @ Rotate(np.pi / 2)
    #             shift = bezier_midpoints[e, :] - midpoint
    #             term_1 += 1 * (np.dot(right_dir, shift) - 0.1) ** 2
    #             term_2 += 3 * np.dot(direction, shift) ** 2
    #         return term_1 + term_2
    #
    #     x0 = bezier_midpoints.flatten() + np.random.randn()
    #     res = minimize(objective, x0=x0, args=(self.rec_edge_dict, self.node_dict), method="SLSQP")
    #     new_bezier_points = res.x.reshape(*true_midpoints.shape)
    #     for e, edge in enumerate(self.rec_edge_dict.keys()):
    #         self.rec_edge_dict[edge]["bezier_point"] = new_bezier_points[e, :]
    #     return None



