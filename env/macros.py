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
IS_ROCK = -99

SENSE_RANGE = 1
MOVE_SPEED = 1

PROB_SENSE = 0.9

MSG_BUFFER_SIZE = 3

WORLD_H = 7
WORLD_W = 7
NUM_AGENTS = 4

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

class Observe(object):
    Quadrant1 = 0
    Quadrant2 = 1
    Quadrant3 = 2
    Quadrant4 = 3

COLORS = ['red', 'green', 'blue', 'black', 'white', 'magenta', 'cyan', 'yellow']

MIN_COLOR = 0
MAX_COLOR = len(COLORS) - 1

## Rewards ##
RWD_STEP_DEFAULT = -0.1
RWD_BUMP_INTO_WALL = -5
RWD_GOAL_FORMATION = 5

# Learning agent
WTS_ACTION_Q = None
WTS_OBSERVE_Q = None

