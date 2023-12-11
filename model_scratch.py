from collections import deque
import datetime
import random
from tqdm import tqdm
import os
import numpy as np

from activation_functions import ActivationReLU, ActivationSoftmax, ActivationLinear
from optimizer import OptimizerAdam
from layers import LayerHidden, LayerConvolutional2D, LayerMaxPooling2D, LayerFlatten, LayerDropout, LayerInput
from accuracy import AccuracyCategorical
from loss import LossMeanSquaredError
from nn import NN
from dataset import scale

from blob_class import Blob, BlobEnv

datetime_format = "%d/%m/%y %H:%M"
model_name = "256x2"

replay_memory_size = 50_000
min_replay_memory_size = 1_000
minibatch_size = 64

discount = 0.99
update_target_every = 5
min_reward_number = -200   # for model save
memory_fraction = 0.20

# env settings
episodes = 1000

# exploration vs exploitation settings
epsilon = 1
epsilon_decay = 0.99975
min_epsilon = 0.001

# stat settings
print_every = 50
show_preview = False

env = BlobEnv()

# For stats
ep_rewards = [-200]

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

print(env.observation_space_values)

class DQNAgent:
    def __init__(self):
        # main model
        self.model = self.create_model()

        # target model
        self.target_model = self.create_model()
        # FUNCTION FOR COPYING WEIGHTS FROM MAIN MODEL NEEDED #

        self.replay_memory = deque()   # aka experience replay. Replay memory is used for more stability and learning from past things. It basically just feeds memory's of states to train the main model of off
        self.target_update_counter = 0   # This will be the number when it has to update the target model and set its weights to the ones of the main model

    def create_model(self):
        model = NN()

        # Layer 1
        model.add(LayerConvolutional2D(kernel_shape=(3, 3, 256), input_shape=env.observation_space_values))
        model.add(ActivationReLU())
        model.add(LayerMaxPooling2D((2, 2), 2))
        model.add(LayerDropout(0.2))

        # Layer 2
        model.add(LayerConvolutional2D(kernel_shape=(3, 3, 256), input_shape=(10, 10, 256)))
        model.add(ActivationReLU())
        model.add(LayerMaxPooling2D((2, 2), 2))
        model.add(LayerDropout(0.2))

        # Layer 3
        model.add(LayerFlatten(256, 128))   # can be tweaked
        model.add(LayerHidden(128, 64))

        # Layer 4
        model.add(LayerHidden(64, env.action_space_size))
        model.add(ActivationLinear())

        model.set_loss_opt_acc(loss=LossMeanSquaredError(), optimizer=OptimizerAdam(learning_rate=0.05, decay=0.001), accuracy=AccuracyCategorical())

        model.finalize()

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(scale(state, 2))[0]

    def train(self, done, step):
        if len(self.replay_memory) < min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.train(X, y, batch_size=minibatch_size)

        if done:
            self.target_update_counter += 1

        if self.target_update_counter > update_target_every:
            # FUNCTION FOR COPYING WEIGHTS FROM MAIN MODEL NEEDED #
            self.target_update_counter = 0


agent = DQNAgent()

# training
for episode in tqdm(range(1, episodes+1), ascii=True, unit="episodes"):
    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space_size)

        new_state, reward, done = env.step(action)
        episode_reward += reward

        if show_preview and not episode % print_every:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)

    if not episode % print_every or episode == 1:
        average_reward = sum(ep_rewards[-print_every:]) / len(ep_rewards[-print_every:])
        min_reward = min(ep_rewards[-print_every:])
        max_reward = max(ep_rewards[-print_every:])

        """# Save model, but only when min reward is greater or equal a set value
        if min_reward >= min_reward_number:
            agent.model.save(f'models/{model_name}--{max_reward:_>7.2f}max--{datetime.datetime.now().strftime(datetime_format)}')"""

    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)

# TODO:
# Fix Conv2D ==> Input image only has 2 dimensions????
# Create function to copy weights from main model to training model
# Fix training (.train used for steps) / build .fit function for nn  (already works?)
# run this program on gpu
