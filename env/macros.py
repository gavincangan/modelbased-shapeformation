import pdb

INVALID = -999
HARD_PLACE = -999

ANY_TIME = -999
SOMETIME = 25
tLIMIT = 25

TWAIT = 0
WAIT_FACTOR = 0.51

MAX_STEPS = 45

UNOCCUPIED = 0
IS_ROCK = -1

SENSE_RANGE = 2
MOVE_SPEED = 1

PROB_SENSE = 0.85

MSG_BUFFER_SIZE = 3

WORLD_H = 9
WORLD_W = 9
NUM_AGENTS = 4

EXPERIMENT_VERSION = 3

FRAME_HEIGHT = 600
FRAME_WIDTH = 600

FRAME_MARGIN = 10
CELL_MARGIN = 5

MAX_AGENTS_IN_CELL = 1



class Actions(object):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    WAIT = 4
    NUM_ACTIONS = 5

class Observe(object):
    Quadrant1 = 0
    Quadrant2 = 1
    Quadrant3 = 2
    Quadrant4 = 3
    NUM_QUADRANTS = 4

COLORS = ['red', 'green', 'blue', 'black', 'white', 'magenta', 'cyan', 'yellow']

MIN_COLOR = 0
MAX_COLOR = len(COLORS) - 1

## Rewards ##
RWD_STEP_DEFAULT = -0.1
RWD_BUMP_INTO_WALL = -2
RWD_GOAL_FORMATION = 2

#agent_act_W7x7_A4_v0

# Learning agent
WTS_ACTION_Q = './save_model/agent_act_W' + str(WORLD_H) + 'x' + str(WORLD_W) + '_A' + str(NUM_AGENTS) + '_v' + str(EXPERIMENT_VERSION) + '.h5'
WTS_OBSERVE_Q = './save_model/agent_obs_W' + str(WORLD_H) + 'x' + str(WORLD_W) + '_A' + str(NUM_AGENTS) + '_v' + str(EXPERIMENT_VERSION)  + '.h5'

WTS_IMWORLD_MODEL = './save_model/agent_imworld_W' + str(WORLD_H) + 'x' + str(WORLD_W) + '_A' + str(NUM_AGENTS) + '_v' + str(EXPERIMENT_VERSION) + '.h5'
WTS_REWARD_MODEL = './save_model/agent_reward_W' + str(WORLD_H) + 'x' + str(WORLD_W) + '_A' + str(NUM_AGENTS) + '_v' + str(EXPERIMENT_VERSION) + '.h5'

'''
Experiment versions:

1. Wrong belief update

'''