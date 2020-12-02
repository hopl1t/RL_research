import numpy as np
import pickle
import gym
import time
import simpleaudio as sa
from threading import Thread
import retro
# from discretizer import Discretizer

H = 200 # number of hidden layer neurons
BATCH_SIZE = 10 # every how many episodes to do a param update?
LEARNING_RATE = 1e-4
GAMMA = 0.99 # discount factor for reward
DECAY_RATE = 0.99 # decay factor for RMSProp leaky sum of grad^2
RESUME = False # RESUME from previous checkpoint?
RENDER = True
PLAY_SOUND = False
RUNS = 3000
MODEL_PICKLE_PATH = 'pong_sound_model.pkl'
REWARED_PICKLE_PATH = 'pong_sound_rewards.pkl'
EPISODE_CIEL = 10 # 8000

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I, pitch=0):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    i = np.zeros((6401))
    i[:-1] = I.astype(np.float).ravel()
    i[-1] = pitch
    return i

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

# audio = []
check_pitch = False
pitch = 0
reward_sums = []
D = (80 * 80) + 1 # input dimensionality: 80x80 grid
if RESUME:
    model = pickle.load(open(MODEL_PICKLE_PATH, 'rb'))
    reward_sums = pickle.load(open(REWARED_PICKLE_PATH, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

# env = retro.make(game='Pong-Atari2600', use_restricted_actions=retro.Actions.DISCRETE)
env = retro.make(game='Pong-Atari2600')
d = Discretizer(env, [['UP'], ['DOWN']])
# d = Discretizer(env, [['RIGHT'], ['LEFT']])
# env = Pong_Discretizer(env)
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

try:
    while episode_number < EPISODE_CIEL:
        if RENDER:
            _ = env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation, pitch)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)

        #       action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
        #       OLD: 2 = RIGHT = UP, 3 = LEFT = DOWN ; NEW: 0 = UP, 1 = UP
        action = 0 if np.random.uniform() < aprob else 1 # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x) # observation
        hs.append(h) # hidden state

        #      y = 1 if action == 2 else 0 # a "fake label"
        y = 1 if action == 0 else 0 # a "fake label"

        dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        #       observation, reward, done, info = env.step(env.action_space.sample())

        observation, reward, done, info = env.step(np.array(d.action(action), dtype=np.int8))

        mono = env.em.get_audio()[:,0]
        pitch = 0
        if check_pitch:
            check_pitch = False
            for i in range(len(mono) - 1):
                if (mono[i] == 0) and (mono[i+1] != 0):
                    for j in range(i+1, len(mono)):
                        if mono[j] == 0:
                            break
                        pitch += 1
                    break
        if mono.any():
            check_pitch = True
        #       audio.append(pitch)

        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs,hs,dlogps,drs = [],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

            # perform rmsprop parameter update every BATCH_SIZE episodes
            if episode_number % BATCH_SIZE == 0:
                for k,v in model.items():
                    g = grad_buffer[k] # gradient
                    rmsprop_cache[k] = DECAY_RATE * rmsprop_cache[k] + (1 - DECAY_RATE) * g**2
                    model[k] += LEARNING_RATE * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sums.append(reward_sum)
            if episode_number % 100 == 0:
                pickle.dump(model, open(MODEL_PICKLE_PATH, 'wb'))
                pickle.dump(reward_sums, open(REWARED_PICKLE_PATH, 'wb'))
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None

            if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                if reward == -1:
                    print ('ep {0}: game finished, reward: {1}'.format(episode_number, reward))
                else:
                    print ('ep {0}: game finished, reward: {1} !!!!!!!!'.format(episode_number, reward))
finally:
    env.close()