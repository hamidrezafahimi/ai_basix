import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../../submodules/SimpleMazeMDP')
from mazemdp.maze_plotter import show_videos
from mazemdp.mdp import Mdp

# For visualization
os.environ["VIDEO_FPS"] = "5"

from mazemdp import create_random_maze

# mdp, nb_states = create_random_maze(10, 10, 0.2)
mdp, nb_states, _, _ = create_random_maze(10, 10, 0.2)

# print(mdp.P)

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


# ----------------- Value Iteration with the V function ----------------------#
# Given a MDP, this algorithm computes the optimal state value function V
# It then derives the optimal policy based on this function
# This function is given


def value_iteration_v(mdp: Mdp, render: bool = True) -> Tuple[np.ndarray, List[float]]:
    # Value Iteration using the state value v
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    v_list = []
    stop = False

    if render:
        mdp.new_render("Value iteration V")

    while not stop:
        v_old = v.copy()
        if render:
            mdp.render(v, title="Value iteration V")

        for x in range(mdp.nb_states):  # for each state x
            # Compute the value of the state x for each action u of the MDP action space
            v_temp = []
            for u in range(mdp.action_space.n):
                if x not in mdp.terminal_states:
                    # Process sum of the values of the neighbouring states
                    summ = 0
                    for y in range(mdp.nb_states):
                        summ = summ + mdp.P[x, u, y] * v_old[y]
                    v_temp.append(mdp.r[x, u] + mdp.gamma * summ)
                else:  # if the state is final, then we only take the reward into account
                    v_temp.append(mdp.r[x, u])

            # Select the highest state value among those computed
            v[x] = np.max(v_temp)

        # Test if convergence has been reached
        if (np.linalg.norm(v - v_old)) < 0.01:
            stop = True
        v_list.append(np.linalg.norm(v))

    if render:
        policy = get_policy_from_v(mdp, v)
        mdp.render(v, policy, title="Value iteration V")

    return v, v_list



v, v_list = value_iteration_v(mdp, render=True)

print("nb_states: ", mdp.nb_states)
print("actions num: ", mdp.action_space.n)
print("V: ", v)
print("terminal_states: ", mdp.terminal_states)
print("P: ", mdp.P)
# show_videos("videos/", prefix="ValueiterationV")
input()