from easy21_env import Easy21

import numpy as np

EPSILON = 0.5
ALPHA = 0.1
GAMMA = 0.1
LAMBDA = 0.5
N_EPS = 100
#train the q value network on the targets

def get_t(feat_range, bin_size, offset):
    tiling = []
    accumulate = feat_range[0] + offset + bin_size
    tiling.append(feat_range[0] + offset)
    while(accumulate < feat_range[1]):
        tiling.append(accumulate)
        accumulate += (bin_size + 1)
    return np.array(tiling)

def clip_val(x, ub):
    lb = 0
    if x < lb:
        return lb
    if x > ub:
        return ub
    return x

def get_tilings():
    #dealer_showing: [1,4][4,7][7,10]
    feat_range = [1,10]
    bin_size = 3
    offset = 0
    a12 = get_t(feat_range, bin_size, offset)

    #player_sum : [1,6][7,12][13,18]
    feat_range = [1,21]
    bin_size = 5
    offset = 0
    a3 = get_t(feat_range, bin_size, offset)

    #player_sum : [4,9][10,15][16,21]
    feat_range = [1,21]
    bin_size = 5
    offset = 3
    a4 = get_t(feat_range, bin_size, offset)

    #actions : [hit, stick]
    a5 = np.array([0,1])
    
    tilings = a12, a3, a4, a5
    return tilings

def get_coding(state, action, tilings):
    #matrix to store the tile coded state:
    t_coding = np.zeros((3,6,2))
    
    #unpack tilings : 
    a12, a3, a4, a5 = tilings
    
    #upper bounds for index clipping
    ub_0 = t_coding.shape[0] - 1
    ub_1 = t_coding.shape[1] - 1
    ub_2 = t_coding.shape[2] - 1
    
    #tile coding:
    #axis_x store the values of which indices of the t_coding matrix to make 1.
    #can be multiple along a tiling because of overlapping tiles
    axis_1 = []
    axis_0 = []
    
    axis_2 = np.digitize(action, a5, right = True)
    
    #multiplied by 2 to get an alternating effect of indicing
    #the a3 tiling will append even indices, and the a4 one will append odd indices
    axis_1.append(clip_val(2 * (np.digitize(state[0],a3, right = True) - 1), ub_1))
    axis_1.append(2 * (np.digitize(state[0],a4) - 1) + 1)
    axis_1 = np.array(axis_1)
    
    axis_0.append(clip_val(np.digitize(state[1], a12, right = False) - 1, ub_0))
    axis_0.append(clip_val(np.digitize(state[1], a12, right = True) - 1, ub_0))
    axis_0 = np.array(axis_0)
    #remove the -1 index that comes for state[1] = 1 in interval [4,9]
    axis_1 = axis_1[axis_1 >= 0]
    # print(axis_0,axis_1,axis_2)
    t_coding[axis_0,axis_1,axis_2] = 1

    return np.reshape(t_coding,(-1))

tilings = get_tilings()

n_inputs = 36

theta = np.random.normal(0, 1 / n_inputs, (1, n_inputs))
elig = np.zeros_like(theta)

env = Easy21()
print(theta)
for i in range(N_EPS):
	states = []
	actions = []
	rewards = []

	state = env.reset()
	done = False

	#select the initial action
	q_vals = []
	for a in range(env.action_space.shape[0]):
			x = get_coding(state, a, tilings)
			q = np.dot(theta, x)
			q_vals.append(q)
	q_vals = np.array(q_vals)
	greedy_a = np.argmax(q_vals)
	#epsilon greedy policy for action selection
	action = np.random.choice(env.action_space)
	selector = np.random.uniform(0,1)
	if selector > EPSILON :
		action = greedy_a
	while not done:
		#execute the action to get next state and reward
		next_state, reward, done = env.step(state,action)
		
		#select optimal action based on value approximation
		q_vals = []
		for a in range(env.action_space.shape[0]):
			x = get_coding(next_state, a, tilings)
			q = np.dot(theta, x)
			q_vals.append(q)
		q_vals = np.array(q_vals)

		greedy_a = np.argmax(q_vals)
		
		q_next_best = np.max(q_vals)
		
		#epsilon greedy policy for action selection
		next_action = np.random.choice(env.action_space)
		selector = np.random.uniform(0,1)
		if selector > EPSILON :
			next_action = greedy_a
		
		x = get_coding(state, action, tilings)
		q_current = np.dot(theta, x)

		td_error = reward + GAMMA * q_next_best - q_current
		elig = GAMMA * LAMBDA * elig + x

		theta = theta - ALPHA * td_error * elig

		states.append(state)
		actions.append(action)
		rewards.append(reward)

		state = next_state
		action = next_action

print(theta)