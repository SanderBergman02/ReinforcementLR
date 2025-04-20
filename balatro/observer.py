import time

def observer(agent):
    # path = "D:\\Steam1\\steamapps\\common\\Balatro\\"
    path = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Balatro\\"
    old_state = ''
    while True:
        state = open(path + "state.txt", "r").read()
        # print(state)
        if old_state != state and state != '':
            old_state = state
            # print(state)
            if state == '1':
                agent.predict_hand()
        time.sleep(0.1)