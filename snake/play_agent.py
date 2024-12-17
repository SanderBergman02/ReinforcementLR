import torch
import random
import numpy as np
from learning_game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
from pathfinding import pathfinding
import time

np.set_printoptions(linewidth=np.inf)
max_mem = 100_000
batch_size = 1000
lr = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9  # <1
        self.memory = deque(maxlen=max_mem)

        self.model = Linear_QNet(13, 256, 3)
        self.model.load_state_dict(torch.load('models/Gen2/model133.pth'))
        self.model.eval()
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,


            # stuck right
            (dir_u and (pathfinding((int(game.h/20)+1, int(game.w/20)+1), game.snake, point_r, game.food, game) == None) and not game.is_collision(point_r)) or
            (dir_d and (pathfinding((int(game.h/20)+1, int(game.w/20)+1), game.snake, point_l, game.food, game) == None) and not game.is_collision(point_l)) or
            (dir_l and (pathfinding((int(game.h/20)+1, int(game.w/20)+1), game.snake, point_u, game.food, game) == None) and not game.is_collision(point_u)) or
            (dir_r and (pathfinding((int(game.h/20)+1, int(game.w/20)+1), game.snake, point_d, game.food, game) == None) and not game.is_collision(point_d)),

            # stuck left
            (dir_d and (pathfinding((int(game.h/20)+1, int(game.w/20)+1), game.snake, point_r, game.food, game) == None) and not game.is_collision(point_r)) or
            (dir_u and (pathfinding((int(game.h/20)+1, int(game.w/20)+1), game.snake, point_l, game.food, game) == None) and not game.is_collision(point_l)) or
            (dir_r and (pathfinding((int(game.h/20)+1, int(game.w/20)+1), game.snake, point_u, game.food, game) == None) and not game.is_collision(point_u)) or
            (dir_l and (pathfinding((int(game.h/20)+1, int(game.w/20)+1), game.snake, point_d, game.food, game) == None) and not game.is_collision(point_d))
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # def train_long_memory(self):
    #     if len(self.memory) > batch_size:
    #         mini_sample = random.sample(self.memory, batch_size)
    #     else:
    #         mini_sample = self.memory
    #
    #     states, actions, rewards, next_states, done = zip(*mini_sample)
    #
    #     self.trainer.train_step(states, actions, rewards, next_states, done)

    # def train_short_memory(self, state, action, reward, next_state, done):
    #     self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        # if random.randint(0, 200) < self.epsilon:
        #     move = random.randint(0, 2)
        #     final_move[move] = 1
        # else:
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move
        reward, done, score = game.play_step(final_move)
        if reward == 0:
            distance = abs(game.food.x/20-game.snake[0].x/20) + abs(game.food.y/20-game.snake[0].y/20)
            # print(distance)
            if (10 - distance) >= 0:
                reward = (10-distance)/2
                # print(reward)
        state_new = agent.get_state(game)

        # train short memory
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        # agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            # agent.n_game += 1
            # agent.train_long_memory()

            # if score > record:
            #     record = score
            #     agent.model.save(file_name='model{}.pth'.format(score))

            print('Game', agent.n_game, 'Score', score, 'Record', record)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_game
            # avg_scores.append(mean_score)
            # plot(plot_scores, avg_scores)

if __name__ == '__main__':
    train()
