from typing import Dict
from torch import Tensor
from melee import Button, Controller
import torch
from Network import TaskNetwork2


class ControllerHelper:
    main_x = .5
    main_y = .5
    c_x = .5
    c_y = .5
    def processMessage(self, message: Dict, controller: Controller):

        if (message["a"] == True):
            controller.press_button(Button.BUTTON_A)
        else:
            controller.release_button(Button.BUTTON_A)
        if (message["b"] == True):
            controller.press_button(Button.BUTTON_B)
        else:
            controller.release_button(Button.BUTTON_B)
        if (message["y"] == True):
            controller.press_button(Button.BUTTON_Y)
        else:
            controller.release_button(Button.BUTTON_Y)

        if (message["z"] == True):
            controller.press_button(Button.BUTTON_Z)
        else:
            controller.release_button(Button.BUTTON_Z)

        controller.tilt_analog(
            Button.BUTTON_MAIN, message["mainStickX"], message["mainStickY"])
        controller.tilt_analog(
            Button.BUTTON_C, message["cStickX"], message["cStickY"])
        controller.press_shoulder(Button.BUTTON_L, message["leftShoulder"])
        if (message["leftShoulder"] >= .9):
            controller.press_button(Button.BUTTON_L)
        else:
            controller.release_button(Button.BUTTON_L)
        # controller.press_shoulder(Button.BUTTON_R, message["rightShoulder"])

        # controller.flush()
    
    def clamp(self, value : float, min_value : float = -4, max_value: float = 4):
        return max(min_value, min(max_value, value))

    def processAnalog(self, analogOutput : Tensor):
        flat_index = torch.argmax(analogOutput)
        max_index = torch.unravel_index(flat_index, analogOutput.shape)
        max_analog_y = max_index[0].item()
        max_analog_x = max_index[1].item()
        shape = analogOutput.shape
        analog = (max_analog_x / (shape[0] - 1), max_analog_y / (shape [1] - 1))
        return analog

