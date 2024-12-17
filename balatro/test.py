from lupa.lua54 import LuaRuntime
import time

def observe_hand():
    f = open("C:\\Program Files (x86)\\Steam\\steamapps\\common\\Balatro\\test.txt", "r")
    print(f.read())
    old_hand = ''
    while True:
        cards_in_hand = open("C:\\Program Files (x86)\\Steam\\steamapps\\common\\Balatro\\test.txt", "r").read()
        if old_hand == cards_in_hand:
            print(cards_in_hand)
            old_hand = cards_in_hand
        data = cards_in_hand.split()
        print(data)
        time.sleep(0.5)

observe_hand()