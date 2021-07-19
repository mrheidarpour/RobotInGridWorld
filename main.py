import numpy as np

from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch

## Include the replay experience
writer = SummaryWriter(comment="-q-learning")

episodes = 10000
gamma = 1 #since it may take several moves to goal, making gamma high

EPSILON_DECAY_LAST_FRAME = 10000
EPSILON_START = 1
EPSILON_FINAL = 0.1

epsilon = EPSILON_START
model = Q_learning(32, [32,  32], 2, hidden_unit)
tgt_model = Q_learning(32, [32,   32], 2, hidden_unit)
#tgt_model.load_state_dict(model.state_dict())
LEARNING_RATE = 1e-3
#optimizer = optim.RMSprop(model.parameters(), lr = 0.00025, momentum = 0.95)
#optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.RMSprop(model.parameters(), lr = LEARNING_RATE)
criterion = torch.nn.MSELoss()
total_rewards = np.array([])
episodes_returns = np.array([])
ave_reward = 0
best_ave_reward = -100
MEAN_REWARD_BOUND = 1
buffer = 800
BATCH_SIZE = 64
memory = ReplayMemory(buffer)   
frame_idx = 0
SYNC_TARGET_FRAMES = 512
solved = 0
for i in range(episodes):
    if solved == 1:
        break
    state = initGrid()
    #state = initGridRand()
    #state =initRandPlayerGrid()

    status = 1
    step = 0
    episode_rewards = np.array([])
    #while game still in progress
    while(status == 1):
        frame_idx +=1
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_model.load_state_dict(model.state_dict())
        if frame_idx % 500 == 0 and LEARNING_RATE > 1e-5:
            LEARNING_RATE = LEARNING_RATE/1.05
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            print('LEARNING_RATE =%s' %LEARNING_RATE )

        v_state = Variable(torch.from_numpy(state)).view(1, -1)

        with torch.no_grad():
            qval = model(v_state)

        if (np.random.random() < epsilon): #choose random action
            action = np.random.randint(0,2)
        else: #choose best action from Q(s,a) values
            action = np.argmax(qval.data)
        #Take action, observe new state S'

        action = filteraction(state, action)


        new_state = makeMove(state, action)
        step +=1
        v_new_state = Variable(torch.from_numpy(new_state)).view(1,-1)
        reward = getReward(new_state)

        state = new_state
        memory.push(v_state.data, action, v_new_state.data, reward)
        if (len(memory) < buffer): #if buffer not filled, add to it
            if reward == TermReward: #if reached terminal state, update game status
                break
            else:
                continue

        #One step reinforcement learning
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
        new_state_batch = Variable(torch.cat(batch.new_state))
        reward_batch = Variable(torch.FloatTensor(batch.reward))
        non_final_mask = (reward_batch != TermReward)
        #Let's run our Q function on S to get Q values for all possible actions
        qval_batch = model(state_batch)
        # we only grad descent on the qval[action], leaving qval[not action] unchanged
        state_action_values = qval_batch.gather(1, action_batch)
        #Get max_Q(S',a)
        with torch.no_grad():
            maxQ = tgt_model(new_state_batch).max(1)[0]
        # maxQ.detach()
#         if reward == -1: #non-terminal state
#             update = (reward + (gamma * maxQ))
#         else: #terminal state
#             update = reward + 0*maxQ
#         y = reward_batch + (reward_batch == -1).float() * gamma *maxQ
        y = reward_batch
        y[non_final_mask] += gamma * maxQ[non_final_mask]
        y = y.view(-1, 1)
        loss = criterion(state_action_values, y)
        print("Game #: %s, loss = %s, epsilon = %s" % (i, loss, epsilon), end='\r')
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.grad.data.clamp_(-1, 1)
        optimizer.step()

        #### visualization ####
        episode_rewards = np.append(episode_rewards, reward)
        if reward == TermReward:
            episodes_returns = np.append(episodes_returns, episode_rewards.sum())
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("loss", loss, frame_idx)

            ave_returns = np.mean(episodes_returns[-10:])
            writer.add_scalar("return_10", ave_returns, frame_idx)
        ######
        ######saveing
        if epsilon <= 0.1 and reward == TermReward and best_ave_reward < ave_returns:
            torch.save(model.state_dict(), "bestModel.dat")
            print("Best reward updated %.3f -> %.3f" % (
                best_ave_reward, ave_returns))
            best_ave_reward = ave_returns
            if best_ave_reward >= MEAN_REWARD_BOUND:
                print("Solved after %d episodes!" % i)
                solved = 1
                break
        ####

        if reward == TermReward:
            status = 0
        if step > 10:
            break
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        # if epsilon > 0.1:
        #     epsilon -= (1 / episodes)

writer.close()
print('frame idx = %d' %(frame_idx))
model.load_state_dict(torch.load('bestModel.dat'))
## Here is the test of AI
def testAlgo(init=0):
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridRand()
    else:
        state = initRandPlayerGrid()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        v_state = Variable(torch.from_numpy(state))
        qval = model(v_state.view(32))
        print(qval)
        action = np.argmax(qval.data) #take action with highest Q-value
        action = filteraction(state,action)
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        print('reward = %s' % reward)
        if reward == TermReward:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break


testAlgo(init=0)
