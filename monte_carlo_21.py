import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import numpy as np
from easy21_env import Easy21


N_EVAL_EPS = 2000
N_0 = 100 #constant for the eps value decay
GAMMA = 1

env = Easy21()

#monte carlo control : q(s,a) using epsilon greedy
N_PLAYER_SUM_VALUES = 21 #1-21
N_DEALER_SHOWING_VALUES = 10 #1-10
N_STATES = N_PLAYER_SUM_VALUES * N_DEALER_SHOWING_VALUES

N_ACTIONS = 2

q = np.zeros((N_STATES, N_ACTIONS))

def map_state(in_state):
	i, j = in_state
	# print('PRE MAPPING : ', i, j)
	map_index = N_DEALER_SHOWING_VALUES * (i - 1) + j - 1
	# print('POST_MAPPING : ', map_index)
	return map_index

N_S = np.zeros((N_STATES,)) #number of times a state has been visited
N_S_A = np.zeros((N_STATES, N_ACTIONS)) #number of times an action has been taken at a state
for i in range(N_EVAL_EPS):
	state = env.reset()
	done = False
	states = []
	rewards = []
	actions = []
	while not done:
		#update no of times state visited
		# print(state)
		N_S[map_state(state)] += 1

		#select action via e greedy:
		eps = N_0 / (N_0 + N_S[map_state(state)])

		#eps greedy policy
		selector = np.random.uniform(0,1)
		if selector > eps:
			#select greedy best action
			action = np.argmax(q[map_state(state)])
		else:
			#select action randomly
			action = np.random.choice(env.action_space)

		#update no of times an action taken at state
		N_S_A[map_state(state), action] += 1

		next_state, reward, done = env.step(state, action)
		states.append(state)
		actions.append(action)
		rewards.append(reward)

		state = next_state

	#computing the return g_t for every state encountered
	running_sum = 0
	g_t = []
	rewards.reverse()

	for reward in rewards:
		running_sum += GAMMA * reward
		g_t.append(running_sum)

	g_t.reverse()

	#updating the q values of every state encountered in episode
	for i in range(len(states)):
		alpha = 1. / N_S_A[map_state(states[i]), actions[i]]
		q[map_state(states[i]), actions[i]] += alpha * (g_t[i] - q[map_state(states[i]), actions[i]])
# print(np.max(q, axis = 1).reshape((21,10)))

with open('./qStarValues.pkl', 'wb') as f:
	pickle.dump(q,f)

print("Dumped True Qs")
x = np.arange(0,21)
y = np.arange(0,10)

X, Y = np.meshgrid(x, y)

Z = np.argmax(q[map_state((X, Y))], axis = 2)
# print(Z.shape)
# exit()
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(X, Y, Z, color='orange')
ax.set_xlabel('PLAYER SUM')
ax.set_ylabel('DEALER SHOWING')
ax.set_zlabel('Q*(s, a)')

plt.show()