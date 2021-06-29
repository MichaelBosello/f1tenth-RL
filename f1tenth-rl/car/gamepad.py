try:
    import inputs
except ImportError:
    pass

from importlib import reload
from threading import Thread
import time

FORWARD_ACTION = 0
RIGHT_ACTION = 1
LEFT_ACTION = 2

class Gamepad():
    def __init__(self):
        self.action = None
        self.dead_man_switch = False
        self.direction = FORWARD_ACTION
        self.autonomous_mode = True
        thread = Gamepad.PadMonitoring(self)
        thread.daemon = True
        thread.start()

    def get_action(self):
        if self.dead_man_switch:
            return self.direction
        else:
            return None
    def is_autonomous_mode(self):
        return self.autonomous_mode

    class PadMonitoring(Thread):
        def __init__(self, gamepad):
            super(Gamepad.PadMonitoring, self).__init__()
            self.gamepad = gamepad
        def run(self):
            while True:
                try:
                    events = inputs.get_gamepad()
                except:
                    self.gamepad.dead_man_switch = False
                    reload(inputs)
                    time.sleep(1)
                    continue
                for event in events:
                    if event.code == "BTN_SOUTH" and event.state == 0:
                        self.gamepad.autonomous_mode = not self.gamepad.autonomous_mode
                    elif event.code == "ABS_RZ" and event.state > 200:
                        self.gamepad.dead_man_switch = True
                    elif event.code == "ABS_RZ" and event.state <= 200:
                        self.gamepad.dead_man_switch = False
                    elif event.code == "ABS_RX" and event.state > 20000:
                        self.gamepad.direction = RIGHT_ACTION
                    elif event.code == "ABS_RX" and event.state < -20000:
                        self.gamepad.direction = LEFT_ACTION
                    elif event.code == "ABS_RX" or event.code == "ABS_Z":
                        self.gamepad.direction = FORWARD_ACTION
                #uncomment this if you need to check the event.code of your gamepad
                #print(event.ev_type, event.code, event.state)



if __name__ == '__main__':
    gamepad = Gamepad()
    while True:
        print("Autonomous mode: {}".format(gamepad.is_autonomous_mode()))
        print("Action: {}".format(gamepad.get_action()))
        time.sleep(1)
