import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../../../submodules/SimpleMazeMDP')
from mazemdp.maze_plotter import show_videos
from mazemdp.mdp import Mdp

# For visualization
os.environ["VIDEO_FPS"] = "5"

from mazemdp import create_random_maze

# mdp, nb_states = create_random_maze(10, 10, 0.2)
mdp, nb_states, _, _ = create_random_maze(10, 10, 0.2)


def get_policy_from_v(mdp: Mdp, v: np.ndarray) -> np.ndarray:
    # Outputs a policy given the state values
    policy = np.zeros(mdp.nb_states)  # initial state values are set to 0
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        v_temp = []
        # for u in mdp.action_space.actions:
        for u in range(mdp.action_space.n):
            if x not in mdp.terminal_states:
                # Process sum of the values of the neighbouring states
                summ = 0
                for y in range(mdp.nb_states):
                    summ = summ + mdp.P[x, u, y] * v[y]
                v_temp.append(mdp.r[x, u] + mdp.gamma * summ)
            else:  # if the state is final, then we only take the reward into account
                v_temp.append(mdp.r[x, u])
        policy[x] = np.argmax(v_temp)
    return policy



# ------------------ Value Iteration with the Q function ---------------------#
# Given a MDP, this algorithm computes the optimal action value function Q
# It then derives the optimal policy based on this function


def value_iteration_q(mdp: Mdp, render: bool = True) -> Tuple[np.ndarray, List[float]]:
    q = np.zeros((mdp.nb_states, mdp.action_space.n))  # initial action values are set to 0
    q_list = []
    stop = False

    if render:
        mdp.new_render("Value iteration Q")

    while not stop:
        qold = q.copy()

        if render:
            mdp.render(q, title="Value iteration Q")

        for x in range(mdp.nb_states):
            for u in range(mdp.action_space.n):
                if x in mdp.terminal_states:
                    q[x, :] = mdp.r[x, u]
                else:
                    summ = 0
                    for y in range(mdp.nb_states):
                        summ += mdp.P[x, u, y] * np.max(qold[y, :])
                    q[x, u] = mdp.r[x, u] + mdp.gamma * summ

        if (np.linalg.norm(q - qold)) <= 0.01:
            stop = True
        q_list.append(np.linalg.norm(q))

    if render:
        mdp.render(q, title="Value iteration Q")
    return q, q_list

q, q_list = value_iteration_q(mdp, render=True)
# show_videos("videos/", prefix="ValueiterationV")
input()