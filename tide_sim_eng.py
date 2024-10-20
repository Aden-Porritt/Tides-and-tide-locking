import numpy as np
from numba.pycc import CC
import numba

cc = CC('tide_sim_eng')

@numba.njit
@cc.export('move_node', 'Tuple((f8[:, :], f8[:, :]))(f8[:, :], f8[:, :], f8[:, :])')
def move_node(nodes_pos, nodes_forces, nodes_velocity):
    nodes_velocity = (nodes_velocity + nodes_forces / 60)
    nodes_pos += nodes_velocity / 60
    return nodes_pos, nodes_velocity

@numba.njit
@cc.export('get_springs_length', 'Tuple((f8[:], f8[:, :]))(f8[:, :], i4[:, :])')
def get_springs_length(nodes_pos, muscles_nodes):
    vectors = np.zeros((len(muscles_nodes), 2))
    for i in range(len(muscles_nodes)):
        vectors[i] = nodes_pos[muscles_nodes[i][0]] - nodes_pos[muscles_nodes[i][1]]
    return np.sqrt(np.sum(vectors ** 2, axis = 1)) + 0.01, vectors

@numba.njit
@cc.export('get_springs_forces', 'Tuple((f8[:, :], f8[:]))(f8[:, :], f8[:], i4[:, :], f8[:], f8[:], f8)')
def get_springs_forces(nodes_pos, start_muscles_length, muscles_nodes, muscles_strength, last_muscles_length, damping):
    muscles_length, unit_vectors = get_springs_length(nodes_pos, muscles_nodes)
    unit_vectors = unit_vectors = unit_vectors / muscles_length.reshape(len(muscles_strength), 1)
    muscles_forces = unit_vectors * ((((start_muscles_length - muscles_length) * muscles_strength * 4) + (last_muscles_length - muscles_length) * damping).reshape(len(muscles_strength), 1))
    nodes_forces = np.zeros((len(nodes_pos), 2))
    for i in range(len(muscles_strength)):
        nodes_forces[muscles_nodes[i][0]] += muscles_forces[i]
        nodes_forces[muscles_nodes[i][1]] -= muscles_forces[i]
    return nodes_forces, muscles_length

@numba.njit
@cc.export('move', 'Tuple((f8[:, :], f8[:, :], f8[:]))(f8[:, :], f8[:, :], f8[:], i4[:, :], f8[:], f8[:], f8, f8, i4)')
def move(nodes_pos, nodes_velocity, start_muscles_length, muscles_nodes, muscles_strength, last_muscles_length, G, damping, fps):
    for _ in range(fps):
        nodes_forces, muscles_length = get_springs_forces(nodes_pos, start_muscles_length, muscles_nodes, muscles_strength, last_muscles_length, damping)
        cen = get_center(nodes_pos)
        for i in range(len(nodes_pos)):
            nodes_forces[i][1] += G * (nodes_pos[i][1] - cen[1])
        nodes_pos, nodes_velocity = move_node(nodes_pos, nodes_forces, nodes_velocity)
        last_muscles_length = np.copy(muscles_length)
    return nodes_pos, nodes_velocity, muscles_length

@numba.njit
@cc.export('get_center', 'f8[:](f8[:, :])')
def get_center(nodes_pos):
    return np.sum(nodes_pos, axis = 0) / len(nodes_pos)

if __name__ == "__main__":
    cc.compile()