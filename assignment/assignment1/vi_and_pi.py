### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters
# Course: CS234 - Assignment 1
# Author: Rindra Ramamonjison
# Student ID: X438181
######################################

import numpy as np
import gym
import time
import itertools
from lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    policy: np.array
        The policy to evaluate. Maps states to actions.
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
        The value function from the given policy.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    old_value = np.zeros(nS)
    for i in range(max_iteration):
        new_value = np.zeros(nS, dtype = 'float')
        for s in range(nS):
            for proba, next_state, reward, _ in P[s][policy[s]]:
                new_value[s] += proba * (reward + gamma * old_value[next_state])
        # print("Value func: {}".format(new_value))

        if converged(old_value, new_value, tol):
            break
        old_value = new_value

    return new_value


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new policy: np.ndarray
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    Q = state_action_value(P, nS, nA, value_from_policy, gamma)
    new_policy = np.argmax(Q, axis = 1)
    return new_policy


def state_action_value(P, nS, nA, value_fn, gamma):
    """
    Return the state-action value matrix associated with a given policy's value function

    :param Ps: dictionary
        for each action a, Ps[a] is a list of outcomes [(proba, next_state, reward, done)]
    :param nA: int
        number of actions
    :param value_from_policy: np.ndarray (nS x 1)
        the value calculated from policy
    :param policy: np.nd
    :param gamma:
    :return:

    """
    Q = np.zeros((nS, nA))
    sa_pairs = ((i,j) for i in range(nS) for j in range(nA))
    for (s, a) in sa_pairs:
        dyn_sa = np.array(P[s][a]) # list of (proba, next_state, reward, done)
        q_value_sa = [proba * (r + gamma * value_fn[int(next_state)]) for proba, next_state, r, _ in dyn_sa]
        Q[s,a] = np.sum(q_value_sa)
    return Q




def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """Runs policy iteration.

    You should use the policy_evaluation and policy_improvement methods to
    implement this method.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
        The optimal value function for each state
    policy: np.ndarray
        An optimal policy
    """
    V = - np.ones((nS,))
    policy = [env.action_space.sample() for i in range(nS)]

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    for i in range(max_iteration):
        print("Policy iteration # {}".format(i))
        value_from_policy = policy_evaluation(P, nS, nA, policy)
        print("\tCurrent Policy: {}".format(policy))
        print("\tValue of current policy: {}".format(value_from_policy))
        new_policy = policy_improvement(P, nS, nA, value_from_policy)
        if converged(value_from_policy, V, tol):
            break

        V = value_from_policy
        policy = new_policy



    return V, policy

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    for i in range(max_iteration):
        print("Value iteration # {}".format(i))
        V_next = np.zeros(nS, dtype = 'float')
        for (s, a) in itertools.product(range(nS), range(nA)):
            q = 0.0
            for proba, next_state, reward, _ in P[s][a]:
                q += proba * (reward + gamma * V[next_state])
            V_next[s] = np.max([V_next[s], q])
        print("\tCurrent value: {}".format(V_next))
        if converged(V, V_next, tol):
            break
        V = V_next

    # Derive policy using optimal value function
    policy = policy_improvement(P, nS, nA, V, gamma)
    return V, policy

def converged(V_prev, V, tol):
    return np.all(np.abs(V - V_prev) < tol)

def example(env):
    """Show an example of gym
    Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
    """
    env.seed(0);
    from gym.spaces import prng; prng.seed(10) # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render();

def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
        Policy: np.array of shape [env.nS]
            The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(1000):
        env.render()
        time.sleep(0.5) # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render();
    print("Total steps: %d " % (t+1))
    print "Episode reward: %f" % episode_reward


def run_single(env, use_value_iteration = True):

    if use_value_iteration:
        print('')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~  Value Iteration  ~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        V, policy = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
        render_single(env, policy)
        print('\nValue Iteration\n\tOptimal Value Function: {}\n\tOptimal Policy: {}'.format(V, policy))
    else:
        print('')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~ Policy Iteration  ~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        V, policy = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=500, tol=1e-3)
        render_single(env, policy)
        print('\nPolicy Iteration\n\tOptimal Value Function: {}\n\tOptimal Policy:{}'.format(V, policy))


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    print('')
    print("##############################")
    print("####  Deterministic-4x4  #####")
    print("##############################")
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    print env.__doc__
    run_single(env, use_value_iteration = True)
    run_single(env, use_value_iteration = False)

    print('')
    print("##############################")
    print("####  Stochastic -4x4  #####")
    print("##############################")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")
    run_single(env, use_value_iteration=True)
    run_single(env, use_value_iteration=False)



