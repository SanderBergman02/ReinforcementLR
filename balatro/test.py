from lupa.lua54 import LuaRuntime
from agents import base_Agent
import time

def make_input(hand):
    list_of_cards = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '6', '7', '8', '9']
    data = [1 if card in hand else 0 for card in list_of_cards]
    return data

def predict_play_discard():
    cards_in_hand = open("D:\\Steam1\\steamapps\\common\\Balatro\\jokers.txt", "r").read()
    hand = cards_in_hand.split()
    data = make_input(hand)
    print(data)
    print(hand)

# state correspond to these activities
# ['PLANET_PACK', '10', 'TAROT_PACK', '9', 'BLIND_SELECT', '7', 'SPLASH', '13',
# 'SPECTRAL_PACK', '15', 'DRAW_TO_HAND', '3', 'HAND_PLAYED', '2', 'ROUND_EVAL', '8',
# 'DEMO_CTA', '16', 'GAME_OVER', '4', 'SANDBOX', '14', 'STANDARD_PACK', '17', 'NEW_ROUND', '19',
# 'BUFFOON_PACK', '18', 'SHOP', '5', 'PLAY_TAROT', '6', 'MENU', '11', 'TUTORIAL', '12', 'SELECTING_HAND', '1']
def observer():
    path = "D:\\Steam1\\steamapps\\common\\Balatro\\"
    agent = base_Agent(path)
    old_state = ''
    while True:
        state = open(path + "state.txt", "r").read()
        print(state)
        if old_state != state and state != '':
            old_state = state
            print(state)
            if state == '1':
                predict_play_discard()
        time.sleep(0.1)

observer()



