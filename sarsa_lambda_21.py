import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import pickle

import numpy as np
from easy21_env import Easy21

LAMBDA = np.arange(0,1.1,0.1)


N_EVAL_EPS = 2000
N_0 = 100 #constant for the eps value decay
GAMMA = 1

env = Easy21()

#monte carlo control : q(s,a) using epsilon greedy
N_PLAYER_SUM_VALUES = 21 #1-21
N_DEALER_SHOWING_VALUES = 10 #1-10
N_STATES = N_PLAYER_SUM_VALUES * N_DEALER_SHOWING_VALUES

N_ACTIONS = 2


def map_state(in_state):
	i, j = in_state
	# print('PRE MAPPING : ', i, j)
	map_index = N_DEALER_SHOWING_VALUES * (i - 1) + j - 1
	# print('POST_MAPPING : ', map_index)
	return map_index

def run_sarsa_lambda(L):
	q = np.zeros((N_STATES, N_ACTIONS))
	N_S = np.zeros((N_STATES,)) #number of times a state has been visited
	N_S_A = np.zeros((N_STATES, N_ACTIONS)) #number of times an action has been taken at a state
	for i in range(N_EVAL_EPS):
		state = env.reset()
		done = False
		states = []
		rewards = []
		actions = []
		elig = np.zeros_like(q)

		#eps greedy policy
		selector = np.random.uniform(0,1)

		eps = N_0 / (N_0 + N_S[map_state(state)])
		if selector > eps:
			#select greedy best action
			action = np.argmax(q[map_state(state)])
		else:
			#select action randomly
			action = np.random.choice(env.action_space)

		while not done:
			#update no of times state visited
			# print(state)
			N_S[map_state(state)] += 1

			#select action via e greedy:
			eps = N_0 / (N_0 + N_S[map_state(state)])

			#update no of times an action taken at state
			N_S_A[map_state(state), action] += 1

			next_state, reward, done = env.step(state, action)

			#pick the next action ::
			selector = np.random.uniform(0,1)
			if selector > eps:
				#select greedy best action from next state's q table lookup
				next_action = np.argmax(q[map_state(next_state)])
			else:
				#select action randomly
				next_action = np.random.choice(env.action_space)

			td_error = reward + q[map_state(next_state), next_action] - q[map_state(state), action]

			elig[map_state(state), action] += 1 #accumulating traces

			for i in range(0, N_PLAYER_SUM_VALUES):
				for j in range(0, N_DEALER_SHOWING_VALUES):
					for a in env.action_space:
						s = (i,j)
						q[map_state(s), a] += GAMMA * elig[map_state(s),a] * td_error
						elig[map_state(s), a] = L * elig[map_state(s), a]

			state = next_state
			action = next_action

			return q

# x = np.arange(0,N_PLAYER_SUM_VALUES)
# y = np.arange(0,N_DEALER_SHOWING_VALUES)

# X, Y = np.meshgrid(x, y)

# Z = np.argmax(q[map_state((X, Y))], axis = 2)
# # print(Z.shape)
# # exit()
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot_wireframe(X, Y, Z, color='blue')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# plt.show()

q_star = []
mses = np.zeros_like(LAMBDA)

with open('./qStarValues.pkl', 'rb') as f:
	q_star = pickle.load(f)

for i in range (LAMBDA.shape[0]):
	print('Running Sarsa Lambda for LAMBDA = ', LAMBDA[i])
	q_l = run_sarsa_lambda(LAMBDA[i])
	mse = np.average((q_star - q_l)**2)
	mses[i] = mse

plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()

ax.plot(LAMBDA, mses)
plt.show()
