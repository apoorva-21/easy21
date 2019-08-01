import numpy as np


class Easy21:

	def __init__(self):
		
		self.SAMPLE_MIN = 1
		self.SAMPLE_MAX = 11
		self.BLACK = 1
		self.RED = -1
		self.DEALER_THRESH = 17
		self.state = (0,0)
		#state representation : (player_sum,dealer_showing)
		#action representation : 0 for hit, 1 for stick (player's actions only, since dealer has fixed policy)
	def draw_card(self, is_reset = False):
		value = int(np.floor(np.random.uniform(self.SAMPLE_MIN, self.SAMPLE_MAX)))
		color_chooser = np.random.uniform(0,1)
		if color_chooser > (1./3) or is_reset:
			color = self.BLACK
		else: 
			color = self.RED
		card = (color, value)
		return card

	def is_bust(self, sum_held):
		if sum_held > 21 or sum_held < 1:
			return True
		return False

	def reset(self):
		c, player_sum = self.draw_card(is_reset = True)
		c, dealer_showing = self.draw_card(is_reset = True)

		self.state = (player_sum, dealer_showing)

		return self.state


	def step(self,in_state, in_action):

		player_sum, dealer_showing = in_state
		reward = 0
		next_state = in_state
		if in_action == 0: # player calls hit

			#draw the card and update player_sum
			color, value = self.draw_card()

			player_sum = player_sum + color * value

			#check for bust etc.
			if self.is_bust(player_sum):
				reward = -1
				player_sum = 0
				dealer_showing = 0 #terminal state
			
		else :
			#player is going to stick, call dealer policy:
			dealer_sum = dealer_showing

			while dealer_sum < self.DEALER_THRESH and not self.is_bust(dealer_sum):
				print('DEALER HITS!')
				#keep hitting until dealer goes bust
				color, value = self.draw_card()
				dealer_sum = dealer_sum + color * value
				print('NEW DEALER SUM = ', dealer_sum)

			if self.is_bust(dealer_sum):
				reward = 1
				player_sum = 0
				dealer_showing = 0 #terminal state
				print('DEALER IS BUST : ', dealer_sum)
			else:
				#give reward based on highest sum:
				if dealer_sum == player_sum:
					reward = 0
				elif dealer_sum < player_sum:
					reward = 1
				elif dealer_sum > player_sum:
					reward = -1
			#terminal state
			player_sum = 0
			dealer_showing = 0

		next_state = (player_sum, dealer_showing)
		return next_state, reward


env = Easy21()
state = env.reset()
print(state)
print(env.step(state, 1))