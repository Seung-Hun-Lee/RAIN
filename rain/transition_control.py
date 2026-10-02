"""Terminate after two confident TC predictions on the final subtask."""
import math


class FinalConditionTCStop:
    def __init__(self):
        self.streak = 0

    def observe(self, value, *, final_condition):
        value = float(value)
        if final_condition and math.isfinite(value) and value >= 0.7:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= 2
