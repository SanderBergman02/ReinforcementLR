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

        #init suits and values
        self.suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        # all jokers, every group in lua is one row here
        self.jokers = [['j_joker', 'j_greedy_joker', 'j_lusty_joker', 'j_wrathful_joker', 'j_gluttenous_joker', 'j_jolly', 'j_zany', 'j_mad', 'j_crazy', 'j_droll', 'j_sly', 'j_wily', 'j_clever', 'j_devious', 'j_crafty'],
                  ['j_half', 'j_stencil', 'j_four_fingers', 'j_mime', 'j_credit_card', 'j_ceremonial', 'j_banner', 'j_mystic_summit', 'j_marble', 'j_loyalty_card', 'j_8_ball', 'j_misprint', 'j_dusk', 'j_raised_fist', 'j_chaos'],
                  ['j_fibonacci', 'j_steel_joker', 'j_scary_face', 'j_abstract', 'j_delayed_grat', 'j_hack', 'j_pareidolia', 'j_gros_michel', 'j_even_steven', 'j_odd_todd', 'j_scholar', 'j_business', 'j_supernova', 'j_ride_the_bus', 'j_space'],
                  ['j_egg', 'j_burglar', 'j_blackboard', 'j_runner', 'j_ice_cream', 'j_dna', 'j_splash', 'j_blue_joker', 'j_sixth_sense', 'j_constellation', 'j_hiker', 'j_faceless', 'j_green_joker', 'j_superposition', 'j_todo_list'],
                  ['j_cavendish', 'j_card_sharp', 'j_red_card', 'j_madness', 'j_square', 'j_seance', 'j_riff_raff', 'j_vampire', 'j_shortcut', 'j_hologram', 'j_vagabond', 'j_baron', 'j_cloud_9', 'j_rocket', 'j_obelisk'],
                  ['j_midas_mask', 'j_luchador', 'j_photograph', 'j_gift', 'j_turtle_bean', 'j_erosion', 'j_reserved_parking', 'j_mail', 'j_to_the_moon', 'j_hallucination', 'j_fortune_teller', 'j_juggler', 'j_drunkard', 'j_stone', 'j_golden'],
                  ['j_lucky_cat', 'j_baseball', 'j_bull', 'j_diet_cola', 'j_trading', 'j_flash', 'j_popcorn', 'j_trousers', 'j_ancient', 'j_ramen', 'j_walkie_talkie', 'j_selzer', 'j_castle', 'j_smiley', 'j_campfire'],
                  ['j_ticket', 'j_mr_bones', 'j_acrobat', 'j_sock_and_buskin', 'j_swashbuckler', 'j_troubadour', 'j_certificate', 'j_smeared', 'j_throwback', 'j_hanging_chad', 'j_rough_gem', 'j_bloodstone', 'j_arrowhead', 'j_onyx_agate', 'j_glass'],
                  ['j_ring_master', 'j_flower_pot', 'j_blueprint', 'j_wee', 'j_merry_andy', 'j_oops', 'j_idol', 'j_seeing_double', 'j_matador', 'j_hit_the_road', 'j_duo', 'j_trio', 'j_family', 'j_order', 'j_tribe'],
                  ['j_stuntman', 'j_invisible', 'j_brainstorm', 'j_satellite', 'j_shoot_the_moon', 'j_drivers_license', 'j_cartomancer', 'j_astronomer', 'j_burnt', 'j_bootstraps', 'j_caino', 'j_triboulet', 'j_yorick', 'j_chicot', 'j_perkeo']]

    def get_hand(self):
        cards_in_hand = open(self.path + "hand.txt", "r").read()
        hand = cards_in_hand.split()
        hand_embed = []
        for line in hand:
            # append the embedding per card to the hand to get a nx4x13 matrix
            hand_embed.append([[1 if suit in line and value in line else 0 for value in self.values] for suit in self.suits])
        return hand_embed

    def get_deck(self):
        cards_in_deck = open(self.path + "deck.txt", "r").read()
        deck = cards_in_deck.split()

        deck_embed = []
        for i in range(52):
            try:
                line = deck[i]
                # append the embedding per card to the hand to get a nx4x13 matrix
                deck_embed.append([[1 if suit in line and value in line else 0 for value in self.values] for suit in self.suits])
            except:
                deck_embed.append([[0 in range(13)] in range(4)])

        return deck_embed

    def get_jokers(self):
        cards_in_hand = open(self.path + "jokers.txt", "r").read()
        joker_list = cards_in_hand.split()

        joker_hand = []
        for i in range(5):
            try:
                # append the embedding per card to the hand to get a nx4x13 matrix
                joker_hand.append([[1 if joker in joker_list[i] else 0 for joker in joker_row] for joker_row in jokers])
            except:
                joker_hand.append([[0 in range(15)] in range(10)])
        return joker_hand

    def get_general_state(self):
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
