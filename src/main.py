import numpy as np
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
from matplotlib import pyplot as plt
from Graph import Graph, Painter
from utils import CustomDistribution
import os

rng = default_rng()
X = CustomDistribution(seed=rng)
Y = X()  # get a frozen version of the distribution
N = 6
density = 3/N
W_rec = 0.8 * random(N, N, density=density, random_state=rng, data_rvs=Y.rvs).toarray()
np.fill_diagonal(W_rec, 0)
W_inp = np.zeros((N, 2))
# W_inp[:3, :3] = np.eye(3)
W_inp[0, 0] = 1
W_inp[2, 1] = 1
# W_inp[3, 2] = 1
W_inp = W_inp + 0.2 * np.random.randn(N, 2)

W_out = np.zeros((3, N))
W_out[:, N-3:] = np.eye(3) + 0.2 * np.random.randn(3,3)
# W_out[1, 4] = 1
G = Graph(W_inp=W_inp, W_rec=W_rec, W_out = W_out)
G.set_nodes()
G.set_edges_from_matrices()
G.set_out_edges_from_matrix()
G.curve_recurrent_connections()
G.painter = Painter()
G.draw_nodes()
G.draw_edges()
plt.tight_layout()
# plt.savefig(f"img/example_{np.random.randint(10000)}.png")
plt.show()