from env.macros import *
from env.gworld import *
from env.visualize import *
from collections import deque
import random
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam

class Sarq:
    def  __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.qmax_nxt = None

class ImaginePath:
    def __init__(self, mod_world, mod_rwd, mod_obs, mod_act, num_steps = 3):
        self.path = []
        self.num_steps = 3
        self.exp_reward = 0
        self.mod_world = mod_world
        self.mod_rwd = mod_rwd
        self.mod_act = mod_act
        self.mod_obs = mod_obs
        self.epsilon = 0.1

    def imagine_state_rwd(self, state_t, action):
        mod_inputs = np.zeros( (2 * WORLD_W * WORLD_H + Actions.NUM_ACTIONS + Observe.TotalOptions, 1) )
        mod_inputs[2 * WORLD_W * WORLD_H, :] = state_t.reshape(2 * WORLD_W * WORLD_H, 1)

        mod_inputs[2 * WORLD_W * WORLD_H + action, :] = 1

        state_tp1 = self.mod_world.predict(mod_inputs, batch_size=1)
        reward = self.mod_rwd.predict(mod_inputs, batch_size=1)
        return ( state_tp1, reward )

    def get_obsseq_tstate(self, tstate, nw_seq = [], seq_len = 5):
        ret = Sarq()
        ret.states.append(tstate)
        curr_step = 0
        obs_quad = 0
        obs_quads = range(Observe.TotalOptions)
        while curr_step < len(nw_seq) and obs_quad < Observe.NUM_QUADRANTS:
            obs_quad = nw_seq[curr_step]
            nxt_state, t_rwd = self.imagine_state_rwd( ret.states[curr_step], Actions.NUM_ACTIONS + obs_quad )
            ret.rewards.append( t_rwd )
            ret.actions.append( obs_quad )
            ret.states.append( nxt_state )
            if (obs_quad in obs_quads):
                obs_quads.remove(obs_quad)
            else:
                print(obs_quad, ' ---- ', obs_quads)
                raise NotImplementedError
            curr_step = curr_step + 1

        while curr_step < seq_len and obs_quad < Observe.NUM_QUADRANTS:
            qval_obs = self.get_qval_obs( ret.states[curr_step] )

            if (random.random() < self.epsilon):
                # obs_quad = random.randint(Observe.Quadrant1, Observe.TotalOptions)
                random.shuffle(obs_quads)
                obs_quad = obs_quads.pop()
            else:
                left_quads = qval_obs[:, np.array(obs_quads)]
                obs_quad_indx = np.argmax(left_quads)
                obs_quad = obs_quads[obs_quad_indx]
                if (obs_quad in obs_quads):
                    obs_quads.remove(obs_quad)
                else:
                    print(obs_quad, ' ---- ', obs_quads)
                    raise NotImplementedError
            nxt_state, t_rwd = self.imagine_state_rwd( ret.states[curr_step], Actions.NUM_ACTIONS + obs_quad)
            ret.rewards.append( t_rwd )
            ret.actions.append( obs_quad )
            ret.states.append( nxt_state )
            curr_step = curr_step + 1

        qval_nxt_act = self.get_qval_act( ret.states[-1] )
        ret.qmax_nxt = np.max(qval_nxt_act)

        return(ret)

    def get_act_seq_state(self, tstate, nw_seq = [], seq_len = 5):
        ret = Sarq()
        ret.states.append(tstate)
        curr_step = 0
        while curr_step < len(nw_seq):
            t_act = nw_seq[curr_step]
            nxt_state, t_rwd = self.imagine_state_rwd(ret.states[curr_step], t_act)
            ret.rewards.append(t_rwd)
            ret.actions.append(t_act)
            ret.states.append(nxt_state)
            curr_step = curr_step + 1

        while curr_step < seq_len:
            qval_act = sa.act_model.predict( ret.states[curr_step], batch_size=1)
            if (random.random() < self.epsilon):
                t_act = np.random.randint(Actions.RIGHT, Actions.WAIT)
            else:
                t_act = (np.argmax(qval_act))
            nxt_state, t_rwd = self.imagine_state_rwd(ret.states[curr_step], t_act)
            ret.rewards.append( t_rwd )
            ret.actions.append( t_act )
            ret.states.append( nxt_state )
            curr_step = curr_step + 1

        qval_nxt_act = self.get_qval_act( ret.states[-1] )
        ret.qmax_nxt = np.max(qval_nxt_act)
        return (ret)

    def get_qval_obs(self, state_t):
        mod_input = state_t.reshape(2 * WORLD_W * WORLD_H, 1)
        qval_obs = self.mod_obs.predict(mod_input, batch_size=1)
        return qval_obs

    def get_qval_act(self, state_t):
        mod_input = state_t.reshape(2 * WORLD_W * WORLD_H, 1)
        qval_act = self.mod_obs.predict(mod_input, batch_size=1)
        return qval_act



class ShapeAgent:
    def __init__(self, show_vis = False):
        self.num_iter = 10000
        self.gamma = 0.975
        self.alpha = 0.85
        self.beta = 0.75
        self.epsilon = 0.1
        self.batchsize = 40
        self.episode_maxlen = 80
        self.replay = deque(maxlen=400)
        self.show_vis = show_vis
        # self.init_env()
        # self.init_model()

    def init_env(self):
        self.env = GridWorld(WORLD_H, WORLD_W)
        bwalls = self.env.get_boundwalls()
        self.env.add_rocks(bwalls)
        self.env.add_agents_rand(NUM_AGENTS)
        self.env.init_agent_beliefs()
        if(self.show_vis):
            self.env.visualize = Visualize(self.env)
            self.env.visualize.draw_world()
            self.env.visualize.draw_agents()
            self.env.visualize.canvas.pack()
            self.disp_update(100)

    def init_model(self):
        shared_model = Sequential()
        shared_model.add(Dense(256, kernel_initializer="lecun_uniform", input_shape=(2 * WORLD_W * WORLD_H,)))
        shared_model.add(Activation('relu'))

        shared_model.add(Dense(256, kernel_initializer="lecun_uniform"))
        shared_model.add(Activation('relu'))
        shared_model.add(Dropout(0.2))

        shared_model.add(Dense(128, kernel_initializer="lecun_uniform"))
        shared_model.add(Activation('relu'))
        shared_model.add(Dropout(0.2))

        act_model = Sequential()
        act_model.add(shared_model)
        act_model.add(Dense(5, kernel_initializer="lecun_uniform", activation='linear'))

        obs_model = Sequential()
        obs_model.add(shared_model)
        obs_model.add(Dense(8, kernel_initializer="lecun_uniform", activation='softmax'))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)

        act_model.compile(adam, 'mse')
        obs_model.compile(adam, 'mse')

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

    def disp_update(self, T = 0):
        self.env.visualize.canvas.update()
        if(T):
            self.env.visualize.canvas.after(T)

if __name__ == "__main__":

    from im_world import ImWorldModel

    sa = ShapeAgent(False)
    sa.init_model()
    sa.load_model()

    im = ImWorldModel()
    im.init_model()
    im.load_model()

    ip = ImaginePath()

    done_step = np.ones(sa.num_iter) * sa.episode_maxlen
    scores = np.zeros(sa.num_iter)

    for i in range(sa.num_iter):

        done_flag = False
        step_count = 0
        sa.init_env()
        agents = sa.env.get_agents()
        shape_reward = False
        rewards = 0
        while(step_count < sa.episode_maxlen and not done_flag):

            random.shuffle(agents)
            step_count += 1
            for agent in agents:
                if not done_flag:
                    # sa.env.visualize.highlight_agent(agent)
                    # obs_quads = range(Observe.TotalOptions)
                    # obs_quad = Observe.Quadrant1

                    state = sa.env.get_agent_state(agent)
                    qval_obs = sa.obs_model.predict(state.reshape(1, 2 * WORLD_H * WORLD_W), batch_size=1)
                    qval_order = np.argsort(qval_obs).tolist()

                    max_seq = None
                    max_q = -10000
                    obs_reward = 0
                    for obs_choice in qval_order[0:4]:
                        t_obs_seq = ip.get_obsseq_tstate( state, [obs_choice] )
                        r_plus_qmax = sum(t_obs_seq.rewards) + t_obs_seq.qmax_nxt
                        if(r_plus_qmax > max_q):
                            max_q = r_plus_qmax
                            max_seq = t_obs_seq

                    for obs_quad in max_seq.actions:
                        if(obs_quad < Observe.NUM_QUADRANTS):
                            sa.env.observe_quadrant(agent, obs_quad)
                            obs_reward = obs_reward + RWD_STEP_DEFAULT
                        else:
                            break

                    state = sa.env.get_agent_state(agent)
                    qval_act = sa.act_model.predict(state, batch_size=1)
                    qval_order = np.argsort(qval_act).tolist()

                    max_seq = None
                    max_q = -10000
                    for act_choice in qval_obs:
                        t_act_seq = ip.get_act_seq_state( state, [act_choice] )
                        r_plus_qmax = sum(t_act_seq.rewards) + t_act_seq.qmax_nxt
                        if(r_plus_qmax > max_q):
                            max_q = r_plus_qmax
                            max_seq = t_act_seq

                    action = max_seq.actions[0]
                    act_reward = sa.env.agent_action(agent, action)

                    closeness_reward = sa.env.check_formation(agent) * RWD_CLOSENESS

                    shape_reward = shape_reward or sa.env.check_shape()

                    rewards = rewards + obs_reward + act_reward + closeness_reward + ( int(shape_reward) * RWD_SHAPE_FORMED )

                    if(shape_reward):
                        done_flag = True
                        done_step[i] = step_count

                    # print ('Agent #%s \tact:%s actQ:%s \n\t\tobs:%s obsQ:%s \n\t\tactR:%s, shapeR:%s' % (agent, action, qval_act, obs_quad, qval_obs, act_reward, shape_reward))
                    # print ('Agent #%s actR:%s, shapeR:%s' % (agent, act_reward, shape_reward))

                    sa.env.share_beliefs(agent)
                    new_state = sa.env.get_agent_state(agent)
        scores[i] = rewards

        print 'Iter:', i, ' scores:', scores[i], ' steps:', done_step[i]
        if(done_step[i] < sa.episode_maxlen):
            print '-- Shaped formed in ', done_step[i], ' steps!'