import random


class Agent:
    def __init__(self, input_size, output_range):
        self._input_size = input_size
        self._output_range = output_range

    def eval(self, agent_input):
        pass


class RandomAgent(Agent):
    def eval(self, agent_input):
        # just be random
        return random.randrange(0, self._output_range)