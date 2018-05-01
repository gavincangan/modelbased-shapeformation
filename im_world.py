from env.macros import *
from env.gworld import *
from env.visualize import *
from collections import deque

from time import time
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard

from demo_vis import ShapeAgent


class ImWorldModel:
    def __init__(self):
        self.num_iter = 10000
        self.gamma = 0.975
        self.epsilon = 0.65
        self.batchsize = 40
        self.episode_maxlen = 1000
        self.replay = deque(maxlen=400)
        # self.init_env()
        # self.init_model()

    def init_model(self):
        reward_model = Sequential()
        reward_model.add(Dense(256, kernel_initializer="lecun_uniform", input_shape=(2 * WORLD_W * WORLD_H + Actions.NUM_ACTIONS + Observe.TotalOptions,)))
        reward_model.add(Activation('relu'))

        reward_model.add(Dense(128, kernel_initializer="lecun_uniform"))
        reward_model.add(Activation('relu'))
        reward_model.add(Dropout(0.2))

        reward_model.add(Dense(64, kernel_initializer="lecun_uniform"))
        reward_model.add(Activation('relu'))
        reward_model.add(Dropout(0.2))

        reward_model.add(Dense(27, kernel_initializer="lecun_uniform", activation='relu'))
        reward_model.add(Dropout(0.2))

        reward_model.add(Dense(1, kernel_initializer="lecun_uniform", activation='linear'))

        imworld_model = Sequential()
        imworld_model.add(Dense(256, kernel_initializer="lecun_uniform", input_shape=(2 * WORLD_W * WORLD_H + Actions.NUM_ACTIONS + Observe.TotalOptions,)))
        imworld_model.add(Activation('relu'))

        imworld_model.add(Dense(128, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(64, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(64, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(128, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(256, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(2 * WORLD_W * WORLD_H, kernel_initializer="lecun_uniform", activation='tanh'))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)

        reward_model.compile(adam, 'mse')
        imworld_model.compile(adam, 'mse')

        self.reward_model = reward_model
        self.imworld_model = imworld_model
        # model.load_weights(WTS_REWARD_MODEL)

    def save_model(self):
        self.imworld_model.save_weights(WTS_IMWORLD_MODEL)
        self.reward_model.save_weights(WTS_REWARD_MODEL)

    def load_model(self):
        if(os.path.isfile(WTS_IMWORLD_MODEL)):
            self.imworld_model.load_weights(WTS_IMWORLD_MODEL)
        if(os.path.isfile(WTS_REWARD_MODEL)):
            self.reward_model.load_weights(WTS_REWARD_MODEL)

class StepMemory:
    def __init__(self):
        self.state = None
        self.action = None
        self.reward = 0
        self.newstate = None
        self.obs_memory = []

if __name__ == "__main__":

    sa = ShapeAgent()
    sa.init_model()
    sa.load_model()

    im = ImWorldModel()
    im.init_model()
    im.load_model()

    tb_act = TensorBoard(log_dir=".logs/act_{}".format(time()))
    tb_obs = TensorBoard(log_dir=".logs/obs_{}".format(time()))
    tb_im_model = TensorBoard(log_dir=".logs/im_model_{}".format(time()))
    tb_im_reward = TensorBoard(log_dir=".logs/im_rwd_{}".format(time()))

    for i in range(sa.num_iter):
        done_flag = False
        step_count = 0
        sa.init_env()
        agents = sa.env.get_agents()
        while(step_count < im.episode_maxlen and not done_flag):
            print '\n>> Count: ', i, ' -- ', step_count
            random.shuffle(agents)
            step_count += 1
            step_mem = dict()
            shape_reward = False
            for agent in agents:
                if not done_flag:
                    print ' #:', agent,
                    step_mem[agent] = StepMemory()
                    obs_quads = range(Observe.TotalOptions)
                    obs_quad = Observe.Quadrant1
                    while obs_quads and obs_quad < Observe.NUM_QUADRANTS:
                        state = sa.env.get_agent_state(agent).reshape(1, 2 * WORLD_H * WORLD_W)
                        qval_obs = sa.obs_model.predict(state, batch_size=1)

                        if(random.random() < im.epsilon):
                            # obs_quad = random.randint(Observe.Quadrant1, Observe.TotalOptions)
                            random.shuffle(obs_quads)
                            obs_quad = obs_quads.pop()
                        else:
                            left_quads = qval_obs[ :, np.array(obs_quads) ]
                            obs_quad_indx = np.argmax( left_quads )
                            obs_quad = obs_quads[obs_quad_indx]
                            if(obs_quad in obs_quads):
                                obs_quads.remove(obs_quad)
                            else:
                                print obs_quad, ' ---- ', obs_quads
                                raise NotImplementedError

                        if(obs_quad < Observe.NUM_QUADRANTS):
                            sa.env.observe_quadrant(agent, obs_quad)
                            print ' O:', obs_quad,
                            new_state = sa.env.get_agent_state(agent).reshape(1, 2 * WORLD_H * WORLD_W)
                            step_mem[agent].obs_memory.append( (state, obs_quad, new_state) )

                    state = sa.env.get_agent_state(agent).reshape(1, 2 * WORLD_H * WORLD_W)
                    qval_act = sa.act_model.predict(state, batch_size=1)

                    if(random.random() < sa.epsilon):
                        action = np.random.randint(Actions.RIGHT, Actions.WAIT)
                    else:
                        action = (np.argmax(qval_act))

                    step_mem[agent].state = state
                    step_mem[agent].action = action
                    print '-- A:', action
                    act_reward = sa.env.agent_action(agent, action)
                    sa.env.share_beliefs(agent)

                    closeness_reward = sa.env.check_formation(agent) * RWD_CLOSENESS

                    shape_reward = shape_reward or sa.env.check_shape()

                    step_mem[agent].reward = act_reward + closeness_reward

                    # print ('Agent #%s \tact:%s actQ:%s \n\t\tobs:%s obsQ:%s \n\t\tactR:%s, shapeR:%s' % (agent, action, qval_act, obs_quad, qval_obs, act_reward, shape_reward))
                    # print ('Agent #%s actR:%s, shapeR:%s' % (agent, closeness_reward))

                    step_mem[agent].newstate = sa.env.get_agent_state(agent).reshape(1, 2 * WORLD_H * WORLD_W)

            for agent in agents:
                my_mem = step_mem[agent]
                sa.replay.append( (my_mem.state, my_mem.action, my_mem.reward, shape_reward, my_mem.newstate) )
                for obs_instance in my_mem.obs_memory:
                    tstate, tobs_quad, tnewstate = obs_instance
                    im.replay.append( (tstate, tobs_quad, my_mem.reward, shape_reward, tnewstate) )
                    im.replay.append( (tstate, Observe.Quadrant4 + random.randint(1,4), my_mem.reward, shape_reward, tstate) )

            if(shape_reward == True):
                print('Count: %s -- Shape formed in %s steps!' %( i, step_count ) )
                done_flag = True

        X_imworld_train = []
        Y_imworld_train = []
        Y_imreward_train = []

        X_obs_train = []
        Y_obs_train = []

        X_act_train = []
        Y_act_train = []

        imworld_train = False

        if (len(sa.replay) > 2 * sa.batchsize):
            minibatch = random.sample(sa.replay, sa.batchsize)
            imworld_train = True

            X_act_train = []
            Y_act_train = []

            for memory in minibatch:

                old_state, action, act_reward, shape_reward, new_state = memory

                old_qval_act = sa.act_model.predict(old_state, batch_size=1)
                new_qval_act = sa.act_model.predict(new_state, batch_size=1)
                max_q_act = np.max(new_qval_act)

                y_act = np.zeros((1, 5))
                y_act[:] = old_qval_act[:]

                y_obs = np.zeros((1, 4))

                if (shape_reward != True):
                    update_reward_act = act_reward + sa.gamma * max_q_act
                else:
                    update_reward_act = act_reward + RWD_SHAPE_FORMED

                old_reward_act = y_act[0][action]

                if (old_reward_act < update_reward_act):
                    y_act[0][action] = sa.alpha * update_reward_act + (1 - sa.alpha) * old_reward_act
                else:
                    y_act[0][action] = sa.beta * update_reward_act + (1 - sa.beta) * old_reward_act

                X_act_train.append(old_state)

                X_imworld_train.append(action)
                Y_imworld_train.append(new_state)
                Y_imreward_train.append( act_reward + (int(shape_reward) * RWD_SHAPE_FORMED) )

                Y_act_train.append(y_act.reshape(Actions.NUM_ACTIONS, ))

            X_act_train = np.array(X_act_train, dtype='float').reshape((-1, 2 * WORLD_H * WORLD_W))
            Y_act_train = np.array(Y_act_train, dtype='float')
            # print('X_act_train: %s\t Y_act_train: %s' % (np.shape(X_act_train), np.shape(Y_act_train)))

            sa.act_model.fit(X_act_train, Y_act_train, batch_size=sa.batchsize, epochs=10, verbose=0, callbacks=[tb_act])
        else:
            imworld_train = False


        if (len(im.replay) > 2 * im.batchsize and imworld_train):
            minibatch = random.sample(im.replay, im.batchsize)

            X_obs_train = []
            Y_obs_train = []
            for memory in minibatch:

                old_state, obs_quad, act_reward, shape_reward, new_state = memory

                old_qval_obs = sa.obs_model.predict(old_state, batch_size=1)
                new_qval_obs = sa.obs_model.predict(new_state, batch_size=1)
                max_q_obs = np.max(new_qval_obs)

                y_obs = np.zeros((1, Observe.TotalOptions))
                y_obs[:] = old_qval_obs[:]

                if (shape_reward != True):
                    update_reward_act = act_reward + sa.gamma * max_q_obs
                else:
                    update_reward_act = act_reward + RWD_SHAPE_FORMED

                old_reward_act = y_obs[0][obs_quad]

                if (old_reward_act < update_reward_act):
                    y_obs[0][obs_quad] = sa.alpha * update_reward_act + (1 - sa.alpha) * old_reward_act
                else:
                    y_obs[0][obs_quad] = sa.beta * update_reward_act + (1 - sa.beta) * old_reward_act

                X_obs_train.append(old_state)

                Y_obs_train.append(y_obs.reshape(1, Observe.TotalOptions))

                X_imworld_train.append( Actions.NUM_ACTIONS + obs_quad )
                Y_imworld_train.append(new_state)
                Y_imreward_train.append( act_reward + (int(shape_reward) * RWD_SHAPE_FORMED) )

            X_obs_train = np.array(X_obs_train, dtype='float').reshape((-1, 2 * WORLD_H * WORLD_W))
            Y_obs_train = np.array(Y_obs_train, dtype='float').reshape((-1, Observe.TotalOptions))
            # print('X_obs_train: %s\t Y_obs_train: %s' % ( np.shape(X_obs_train), np.shape(Y_obs_train) ) )

            sa.obs_model.fit(X_obs_train, Y_obs_train, batch_size=sa.batchsize, epochs=10, verbose=0, callbacks=[tb_obs])
        else:
            imworld_train = False

        if imworld_train:
            imworld_train = False

            # print len(X_imworld_train)
            # print X_imworld_train[0]
            num_samples = len(X_imworld_train)
            action_array = np.array(X_imworld_train)
            X_imworld_train = np.zeros( (num_samples, Actions.NUM_ACTIONS + Observe.TotalOptions) )
            X_imworld_train[np.arange(num_samples), action_array] = 1

            # print('Concat -- X_act_train: %s\t X_obs_train: %s' % (np.shape(X_act_train), np.shape(X_obs_train)))
            X_oldstate_train = np.concatenate( (X_act_train, X_obs_train), 0 )
            # print('Concat -- X_imworld_train: %s\t X_oldstate_train: %s' % (np.shape(X_imworld_train), np.shape(X_oldstate_train)))
            X_imworld_train = np.concatenate( (X_oldstate_train, X_imworld_train), 1 )


            Y_imworld_train = np.array(Y_imworld_train).reshape((-1, 2 * WORLD_H * WORLD_W))
            Y_imreward_train = np.array(Y_imreward_train).reshape((-1, 1))
            # print('Y_imreward_train: %s\t Y_imworld_train: %s\t Y_imreward_train: %s' % (np.shape(X_imworld_train), np.shape(Y_imworld_train), np.shape(Y_imreward_train)))
            im.imworld_model.fit(X_imworld_train, Y_imworld_train, batch_size=sa.batchsize, epochs=10, verbose=0, callbacks=[tb_im_model])
            im.reward_model.fit(X_imworld_train, Y_imreward_train, batch_size=sa.batchsize, epochs=10, verbose=0, callbacks=[tb_im_reward])

        if (sa.epsilon > 0.1):
            sa.epsilon -= (0.01)
        if (im.epsilon > 0.1):
            im.epsilon -= (0.01)

        sa.save_model()
        im.save_model()

