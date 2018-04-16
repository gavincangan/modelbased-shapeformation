from env.macros import *
from env.gworld import *
from env.visualize import *
from collections import deque
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

class DemoShapeAgent:
    def __init__(self):
        self.num_iter = 10000
        self.gamma = 0.975
        self.epsilon = 0.25
        self.batchsize = 40
        self.episode_maxlen = 80
        self.replay = deque(maxlen=2000)
        # self.init_env()
        # self.init_model()

    def init_env(self):
        self.env = GridWorld(WORLD_H, WORLD_W)
        bwalls = self.env.get_boundwalls()
        self.env.add_rocks(bwalls)
        self.env.add_agents_rand(NUM_AGENTS)
        self.env.visualize = Visualize(self.env)
        self.env.visualize.draw_world()
        self.env.visualize.draw_agents()
        self.env.visualize.canvas.pack()
        self.disp_update(100)

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
        self.act_model.save_weights('./save_model/agent_act_W7x7_A4_v0.h5')
        self.obs_model.save_weights('./save_model/agent_obs_W7x7_A4_v0.h5')

    def load_model(self):
        self.act_model.load_weights('./save_model/agent_act_W7x7_A4_v0.h5')
        self.obs_model.load_weights('./save_model/agent_obs_W7x7_A4_v0.h5')

    def disp_update(self, T = 0):
        self.env.visualize.canvas.update()
        if(T):
            self.env.visualize.canvas.after(T)

if __name__ == "__main__":

    sa = DemoShapeAgent()
    sa.init_model()
    sa.load_model()
    for i in range(sa.num_iter):

        done_flag = False
        step_count = 0
        sa.init_env()
        agents = sa.env.get_agents()
        while(step_count < sa.episode_maxlen and not done_flag):

            random.shuffle(agents)
            step_count += 1
            for agent in agents:
                if not done_flag:
                    sa.env.visualize.highlight_agent(agent)
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

                    print ('Agent #%s \tact:%s actQ:%s \n\t\tobs:%s obsQ:%s \n\t\tactR:%s, shapeR:%s' % (agent, action, qval_act, obs_quad, qval_obs, act_reward, shape_reward))

                    new_state = sa.env.get_agent_state(agent)

                    sa.disp_update(1000)
        sa.disp_update(2500)



