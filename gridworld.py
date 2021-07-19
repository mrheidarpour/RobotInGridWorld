import numpy as np

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

def filteraction(state,action):
    for i in range(0,4):
        if state[0,3,i] ==1:
            return 1
        if state[0,i,3] == 1:
            return 0
    return action

#Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((2,4,4))
    #place player
    state[0, 0, 0] = 1
    #place coins
    state[1, 1, 1] = 1
    state[1, 2, 2] = 1
    state[1, 2, 3] = 1
    return state

def initRandPlayerGrid():
    state = np.zeros((2,4,4))
    #place player
    p = randPair(0, 4)
    state[0][p] = 1
    # place coins
    state[1, 1, 1] = 1
    state[1, 2, 2] = 1
    state[1, 2, 3] = 1
    if p == (1, 1) or p == (2, 2) or p == (2, 3):
        return initRandPlayerGrid()

    return state

#Initialize grid so that coins are all randomly placed
def initGridRand():
    state = np.zeros((2, 4, 4))
    #place player
    state[0][0, 0] = 1
    #place coins
    #Ncoins = np.random.randint(1, 5)
    Ncoins = 3
    ncoins = Ncoins
    while ncoins > 0:
        c = randPair(0, 4)
        if c != (0, 0):
            state[1][c] = 1
            ncoins -= 1

    if (np.sum(state)!= Ncoins+1):
        #print('Invalid grid. Rebuilding..')
        return initGridRand()
    
    return state


def makeMove(state, action):
    assert action == 0 or action == 1
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = getLoc(state, 0)
    #state = np.zeros((4,4,4))
    state[0][player_loc] = 0
    actions = [[1,0], [0,1]]
    #e.g. down => (player row + 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
        state[0][new_loc] = 1
    else:
        state[0][player_loc] = 1

    return state

def getLoc(state, level):
    for i in range(0,4):
        for j in range(0,4):
            if (state[level][i,j] == 1):
                return i,j

TermReward = 0
def getReward(state):
    player_loc = getLoc(state, 0)
    if player_loc[0] == 3 and player_loc[1] == 3:
        return TermReward
    if state[1][player_loc] == 1:
        state[1][player_loc] = -1
        return 1
    else:
        return -1
    
def dispGrid(state):
    grid = np.zeros((4,4), dtype= str)
    player_loc = getLoc(state, 0)
    for i in range(0,4):
        for j in range(0,4):
            if state[1][i,j] ==1:
                grid[i,j] = 'C'
            elif state[1][i,j] ==-1:
                grid[i,j] = '*'
            else:
                grid[i,j] = ' '
    grid[player_loc] = 'P'
    return grid