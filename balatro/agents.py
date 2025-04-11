import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, Linear_QNetGen3, QTrainer

import time
np.set_printoptions(linewidth=np.inf)
max_mem = 100_000
batch_size = 1000
lr = 0.001

class base_Agent:
    def __init__(self, path):
        self.path = path
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9  # <1
        self.memory = deque(maxlen=max_mem)

        self.model = Linear_QNetGen3(13, 128, 3)
        # self.model.load_state_dict(torch.load('./models/model65.pth'))
        # self.model.eval()
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)

    def get_hand(self):
        hand = open(self.path + "deck.txt", "r").read()
        hand_embed = []
        list_of_cards = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '6', '7', '8', '9']
        for i in range(10):
            if i < len(hand):
                data = [1 if card in hand[i] else 0 for card in list_of_cards]
            else:
                data = [0 for card in list_of_cards]
            hand_embed.append(data)
        return hand_embed

    def get_deck(self):
        deck = open(self.path + "hand.txt", "r").read()
        deck_embed = []
        list_of_cards = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '6', '7', '8', '9']
        for i in range(10):
            if i < len(deck):
                data = [1 if card in deck[i] else 0 for card in list_of_cards]
            else:
                data = [0 for card in list_of_cards]
            deck_embed.append(data)
        return deck_embed

    def get_jokers(self):
        jokers = open(self.path + "jokers.txt", "r").read()
        jokers_embed = []
        list_of_cards = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '6', '7', '8', '9']
        for i in range(10):
            if i < len(jokers):
                data = [1 if card in jokers[i] else 0 for card in list_of_cards]
            else:
                data = [0 for joker in list_of_cards]
            jokers_embed.append(data)
        return jokers_embed

    def get_general_state(self, game):
        return 1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Adjust for new states etc
    def train_long_memory(self):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, done = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
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
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(file_name='model{}.pth'.format(score))

            print('Game', agent.n_game, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            avg_scores.append(mean_score)
            plot(plot_scores, avg_scores)

if __name__ == '__main__':
    train()
