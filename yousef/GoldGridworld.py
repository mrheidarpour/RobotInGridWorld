import torch
import numpy as np


class GoldGridworld:
    def __init__(self, size=4, mode='static', num_coins=5):
        if size < 4:
            print("Minimum board size is 4. Initialized to size 4.")
            size = 4

        self.size = size
        self.py = 0
        self.px = 0
        # initializing the board:
        self.board = np.full((size, size), ' ')
        self.board[0, 0] = 'P'

        if mode == 'static':
            self.positions = list(np.arange(1, 1 + num_coins))
        else:
            self.positions = list(
                np.random.choice(np.arange(1, size**2 - 1),
                                 num_coins,
                                 replace=False))

        for pos in self.positions:
            self.board[pos // size, pos % size] = '$'

    def get_state(self):
        p = torch.zeros(self.size, self.size, dtype=torch.float32)
        p[self.px, self.py] = 1

        c = torch.zeros(self.size**2, dtype=torch.float32)
        c[self.positions] = 1
        c = c.view(self.size, self.size)

        return torch.stack([p, c]).view(1, -1)

    def is_finished(self):
        if self.board[self.size - 1, self.size - 1] == 'P':
            return True
        return False

    def move(self, action):
        assert action == 'r' or action == 'd'
        reward = 0
        if self.py == self.size - 1 and action == 'r':
            return reward
        if self.px == self.size - 1 and action == 'd':
            return reward

        self.board[self.px, self.py] = ' '

        if action == 'r':
            self.py += 1
        else:
            self.px += 1
        if self.board[self.px, self.py] == '$':
            self.positions.remove(self.px * self.size + self.py)
            reward = 1

        self.board[self.px, self.py] = 'P'
        return reward

    def display(self):
        print(self.board)