from lupa.lua54 import LuaRuntime
from agents import base_Agent
import time

def make_input(hand):
    list_of_cards = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '6', '7', '8', '9']
    data = [1 if card in hand else 0 for card in list_of_cards]
    return data

def predict_play_discard(path):
    cards_in_hand = open(path+"jokers.txt", "r").read()
    joker_list = cards_in_hand.split()
    # suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    # values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
    #all jokers, every group in lua is one row here
    jokers = [['j_joker', 'j_greedy_joker', 'j_lusty_joker', 'j_wrathful_joker', 'j_gluttenous_joker', 'j_jolly', 'j_zany', 'j_mad', 'j_crazy', 'j_droll', 'j_sly', 'j_wily', 'j_clever', 'j_devious', 'j_crafty'],
              ['j_half', 'j_stencil', 'j_four_fingers', 'j_mime', 'j_credit_card', 'j_ceremonial', 'j_banner', 'j_mystic_summit', 'j_marble', 'j_loyalty_card', 'j_8_ball', 'j_misprint', 'j_dusk', 'j_raised_fist', 'j_chaos'],
              ['j_fibonacci', 'j_steel_joker', 'j_scary_face', 'j_abstract', 'j_delayed_grat', 'j_hack', 'j_pareidolia', 'j_gros_michel', 'j_even_steven', 'j_odd_todd', 'j_scholar', 'j_business', 'j_supernova', 'j_ride_the_bus', 'j_space'],
              ['j_egg', 'j_burglar', 'j_blackboard', 'j_runner', 'j_ice_cream', 'j_dna', 'j_splash', 'j_blue_joker', 'j_sixth_sense', 'j_constellation', 'j_hiker', 'j_faceless', 'j_green_joker', 'j_superposition', 'j_todo_list'],
              ['j_cavendish', 'j_card_sharp', 'j_red_card', 'j_madness', 'j_square', 'j_seance', 'j_riff_raff', 'j_vampire', 'j_shortcut', 'j_hologram', 'j_vagabond', 'j_baron', 'j_cloud_9', 'j_rocket', 'j_obelisk'],
              ['j_midas_mask', 'j_luchador', 'j_photograph', 'j_gift', 'j_turtle_bean', 'j_erosion', 'j_reserved_parking', 'j_mail' ,'j_to_the_moon', 'j_hallucination', 'j_fortune_teller', 'j_juggler', 'j_drunkard', 'j_stone', 'j_golden'],
              ['j_lucky_cat', 'j_baseball', 'j_bull', 'j_diet_cola', 'j_trading', 'j_flash', 'j_popcorn', 'j_trousers', 'j_ancient', 'j_ramen', 'j_walkie_talkie', 'j_selzer', 'j_castle', 'j_smiley', 'j_campfire'],
              ['j_ticket', 'j_mr_bones', 'j_acrobat', 'j_sock_and_buskin', 'j_swashbuckler', 'j_troubadour', 'j_certificate', 'j_smeared', 'j_throwback', 'j_hanging_chad', 'j_rough_gem', 'j_bloodstone', 'j_arrowhead', 'j_onyx_agate', 'j_glass'],
              ['j_ring_master', 'j_flower_pot', 'j_blueprint', 'j_wee', 'j_merry_andy', 'j_oops', 'j_idol', 'j_seeing_double', 'j_matador', 'j_hit_the_road', 'j_duo', 'j_trio', 'j_family', 'j_order', 'j_tribe'],
              ['j_stuntman', 'j_invisible', 'j_brainstorm', 'j_satellite', 'j_shoot_the_moon', 'j_drivers_license', 'j_cartomancer', 'j_astronomer', 'j_burnt', 'j_bootstraps', 'j_caino', 'j_triboulet', 'j_yorick', 'j_chicot', 'j_perkeo']]
    joker_hand = []
    for i in range(5):
        try:
            #append the embedding per card to the hand to get a nx4x13 matrix
            joker_hand.append([[1 if joker in joker_list[i] else 0 for joker in joker_row] for joker_row in jokers])
        except:
            joker_hand.append([[0 in range(15)] in range(10)])
    print(joker_list)
    print(joker_hand)
    return joker_hand

# state correspond to these activities
# ['PLANET_PACK', '10', 'TAROT_PACK', '9', 'BLIND_SELECT', '7', 'SPLASH', '13',
# 'SPECTRAL_PACK', '15', 'DRAW_TO_HAND', '3', 'HAND_PLAYED', '2', 'ROUND_EVAL', '8',
# 'DEMO_CTA', '16', 'GAME_OVER', '4', 'SANDBOX', '14', 'STANDARD_PACK', '17', 'NEW_ROUND', '19',
# 'BUFFOON_PACK', '18', 'SHOP', '5', 'PLAY_TAROT', '6', 'MENU', '11', 'TUTORIAL', '12', 'SELECTING_HAND', '1']
def observer():
    # path = "D:\\Steam1\\steamapps\\common\\Balatro\\"
    path = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Balatro\\"
    agent = base_Agent(path)
    old_state = ''
    while True:
        state = open(path + "state.txt", "r").read()
        # print(state)
        if old_state != state and state != '':
            old_state = state
            # print(state)
            if state == '1':
                predict_play_discard(path)
        time.sleep(0.1)

observer()



