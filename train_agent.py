from env.macros import *
from env.gworld import *
from env.visualize import *
from collections import deque
import random
import os.path

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

class ShapeAgent:
    def __init__(self):
        self.num_iter = 10000
        self.gamma = 0.975
        self.epsilon = 1
        self.batchsize = 400
        self.episode_maxlen = 80
        self.alpha = 0.85
        self.beta = 0.45
        self.replay = deque(maxlen=2000)
        self.init_env()
        self.init_model()

    def init_env(self):
        self.env = GridWorld(WORLD_H, WORLD_W)
        self.env.add_agents_rand(NUM_AGENTS)

    def init_model(self):
        shared_model = Sequential()
        shared_model.add(Dense(164, init='lecun_uniform', input_shape=(2 * WORLD_W * WORLD_H,)))
        shared_model.add(Activation('relu'))

        shared_model.add(Dense(150, init='lecun_uniform'))
        shared_model.add(Activation('relu'))
        # shared_model.add(Dropout(0.2))

        act_model = Sequential()
        act_model.add(shared_model)
        act_model.add(Dense(5, init='lecun_uniform', activation='softmax'))

        obs_model = Sequential()
        obs_model.add(shared_model)
        obs_model.add(Dense(4, init='lecun_uniform', activation='softmax'))

        rms = RMSprop()

        act_model.compile(rms, "categorical_crossentropy")
        obs_model.compile(rms, "mse")

        self.act_model = act_model
        self.obs_model = obs_model
        # model.load_weights(WTS_ACTION_Q)

    def save_model(self):
        self.act_model.save_weights(WTS_ACTION_Q)
        self.obs_model.save_weights(WTS_OBSERVE_Q)

    def load_model(self):
        if(os.path.isfile(WTS_ACTION_Q)):
            self.act_model.load_weights(WTS_ACTION_Q)
        if (os.path.isfile(WTS_OBSERVE_Q)):
            self.obs_model.load_weights(WTS_OBSERVE_Q)

if __name__ == "__main__":

    sa = ShapeAgent()
    sa.init_model()
    sa.load_model()
    for i in range(sa.num_iter):

        step_count = 0
        sa.init_env()
        agents = sa.env.get_agents()
        print("Game #:%s" % (i,))
        while(step_count < sa.episode_maxlen):
            # print step_count
            random.shuffle(agents)
            step_count += 1
            for agent in agents:

                state = sa.env.get_agent_state(agent)
                qval_act = sa.act_model.predict(state.reshape(1, 2 * WORLD_H * WORLD_W), batch_size=1)
                qval_obs = sa.obs_model.predict(state.reshape(1, 2 * WORLD_H * WORLD_W), batch_size=1)

                if(random.random() < sa.epsilon):
                    action = np.random.randint(Actions.RIGHT, Actions.WAIT)
                    obs_quad = random.randint(Observe.Quadrant1, Observe.Quadrant4)
                else:
                    action = (np.argmax(qval_act))
                    obs_quad = (np.argmax(qval_obs))

                sa.env.observe_quadrant(agent, obs_quad)
                act_reward = sa.env.agent_action(agent, action)
                shape_reward = sa.env.check_formation(agent) * RWD_GOAL_FORMATION

                new_state = sa.env.get_agent_state(agent)

                if(step_count % 40 == 0):
                    print('Agent #%s \tact:%s actQ:%s \n\t\tobs:%s obsQ:%s \n\t\tactR:%s, shapeR:%s' % ( \
                    agent, action, qval_act, obs_quad, qval_obs, act_reward, shape_reward))

                sa.replay.append( (state, action, obs_quad, act_reward, shape_reward, new_state) )

            if(len(sa.replay) > 2 * sa.batchsize):
                minibatch = random.sample(sa.replay, sa.batchsize)
                X_train = []
                Y_act_train = []
                Y_obs_train = []

                for memory in minibatch:
                    old_state, action, obs_quad, act_reward, shape_reward, new_state = memory

                    old_qval_act = sa.act_model.predict(old_state.reshape(1, 2 * WORLD_H * WORLD_W), batch_size=1)
                    old_qval_obs = sa.obs_model.predict(old_state.reshape(1, 2 * WORLD_H * WORLD_W), batch_size=1)

                    new_qval_act = sa.act_model.predict(new_state.reshape(1, 2 * WORLD_H * WORLD_W), batch_size=1)
                    new_qval_obs = sa.obs_model.predict(new_state.reshape(1, 2 * WORLD_H * WORLD_W), batch_size=1)

                    max_q_act = np.max(new_qval_act)
                    max_q_obs = np.max(new_qval_obs)

                    y_act = np.zeros((1, 5))
                    y_act[:] = old_qval_act[:]

                    y_obs = np.zeros((1, 4))
                    y_obs[:] = old_qval_obs[:]

                    if(shape_reward < 50):
                        update_reward_act = act_reward + shape_reward + sa.gamma * max_q_act
                        update_reward_obs = act_reward + shape_reward + sa.gamma * max_q_obs
                    else:
                        update_reward_act = act_reward + shape_reward
                        update_reward_obs = act_reward + shape_reward

                    old_reward_act = y_act[0][action]
                    old_reward_obs = y_obs[0][obs_quad]

                    if(old_reward_act < update_reward_act):
                        y_act[0][action] = sa.alpha * update_reward_act + (1 - sa.alpha) * old_reward_act
                    else:
                        y_act[0][action] = sa.beta * update_reward_act + (1 - sa.beta) * old_reward_act

                    if (old_reward_obs < update_reward_obs):
                        y_obs[0][obs_quad] = sa.alpha * update_reward_obs + (1 - sa.alpha) * old_reward_obs
                    else:
                        y_obs[0][obs_quad] = sa.beta * update_reward_obs + (1 - sa.beta) * old_reward_obs


                    X_train.append(old_state.reshape(2 * WORLD_H * WORLD_W, ))

                    Y_act_train.append(y_act.reshape(5,))
                    Y_obs_train.append(y_obs.reshape(4, ))


                X_train = np.array(X_train, dtype='float')
                Y_act_train = np.array(Y_act_train, dtype='float')
                Y_obs_train = np.array(Y_obs_train, dtype='float')
                sa.act_model.fit(X_train, Y_act_train, batch_size=sa.batchsize, epochs=5, verbose=0 )

        if(sa.epsilon > 0.1):
            sa.epsilon -= (1/1000)
        if(i % 5 == 0):
            sa.save_model()


