import argparse
from collections import deque
from enum import Enum
import json

from multiprocessing import Process, Queue, Manager
import signal
import sys
import time

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import requests

import melee
from melee.controller import Controller
from melee.console import Console, GameState, PlayerState
from torch import Tensor, tensor
import torch

from Network import (
    CPPNConnectionQuery,
    DefaultActivationFunctionMapper,
    NetworkBuilder,
    NetworkProcessorFactory,
    Substrate,
    TaskNetwork2,
    json_to_network_genome,
)
from NetworkGenome import NetworkGenome
from smash.ControllerHelper import ControllerHelper
from smash.cpuSelector import choose_character

def excluded_actions():
    return [
            melee.Action.SHIELD_BREAK_FALL.value,
            melee.Action.SHIELD_BREAK_DOWN_D.value,
            melee.Action.SHIELD_BREAK_DOWN_U.value,
            melee.Action.SHIELD_BREAK_TEETER.value,
            melee.Action.SHIELD_BREAK_FLY.value,
            melee.Action.SHIELD_BREAK_STAND_D.value,
            melee.Action.SHIELD_BREAK_STAND_U.value,
            melee.Action.CROUCH_START.value,
            melee.Action.CROUCH_END.value,
            melee.Action.GROUND_ROLL_SPOT_DOWN.value,
            melee.Action.GROUND_SPOT_UP.value,
            melee.Action.DAMAGE_AIR_1.value,
            melee.Action.DAMAGE_AIR_2.value,
            melee.Action.DAMAGE_AIR_3.value,
            melee.Action.REBOUND.value,
            melee.Action.REBOUND_STOP.value,
            melee.Action.LANDING_SPECIAL.value,
            melee.Action.SHIELD_STUN.value,
            melee.Action.DAMAGE_FLY_HIGH.value,
            melee.Action.DAMAGE_FLY_LOW.value,
            melee.Action.DAMAGE_FLY_NEUTRAL.value,
            melee.Action.DAMAGE_FLY_ROLL.value,
            melee.Action.DAMAGE_FLY_TOP.value,
            melee.Action.DAMAGE_GROUND.value,
            melee.Action.DAMAGE_HIGH_1.value,
            melee.Action.DAMAGE_HIGH_2.value,
            melee.Action.DAMAGE_HIGH_3.value,
            melee.Action.DAMAGE_ICE.value,
            melee.Action.DAMAGE_ICE_JUMP.value,
            melee.Action.DAMAGE_LOW_1.value,
            melee.Action.DAMAGE_LOW_2.value,
            melee.Action.DAMAGE_LOW_3.value,
            melee.Action.DAMAGE_NEUTRAL_1.value,
            melee.Action.DAMAGE_NEUTRAL_2.value,
            melee.Action.DAMAGE_NEUTRAL_3.value,
            melee.Action.DAMAGE_SCREW.value,
            melee.Action.DAMAGE_SCREW_AIR.value,
            melee.Action.GRABBED.value,
            melee.Action.GRABBED_WAIT_HIGH.value,
            melee.Action.GRAB_PUMMELED.value,
            melee.Action.LYING_GROUND_DOWN.value,
            melee.Action.LYING_GROUND_UP_HIT.value,
            melee.Action.LYING_GROUND_UP.value,
            melee.Action.FALLING.value,
            melee.Action.ON_HALO_DESCENT.value,
            melee.Action.ON_HALO_WAIT.value,
            melee.Action.THROWN_BACK.value,
            melee.Action.THROWN_F_HIGH.value,
            melee.Action.THROWN_F_LOW.value,
            melee.Action.THROWN_DOWN.value,
            melee.Action.THROWN_DOWN_2.value,
            melee.Action.THROWN_FB.value,
            melee.Action.THROWN_FF.value,
            melee.Action.THROWN_UP.value,
            melee.Action.THROWN_FORWARD.value,
            melee.Action.TUMBLING.value,
            melee.Action.SHIELD_START.value,
            melee.Action.SHIELD_RELEASE.value,
            melee.Action.LOOPING_ATTACK_MIDDLE.value,
            melee.Action.LOOPING_ATTACK_END.value,
            melee.Action.LANDING.value,
            melee.Action.FAIR_LANDING.value,
            melee.Action.BAIR_LANDING.value,
            melee.Action.LANDING_SPECIAL.value,
            melee.Action.NAIR_LANDING.value,
            melee.Action.DAIR_LANDING.value,
            melee.Action.UAIR_LANDING.value,
            melee.Action.DEAD_FALL.value,
            melee.Action.FALLING.value,
            melee.Action.FALLING_BACKWARD.value,
            melee.Action.FALLING_BACKWARD.value,
            melee.Action.SPECIAL_FALL_BACK.value,
            melee.Action.SPECIAL_FALL_FORWARD.value,
            melee.Action.EDGE_JUMP_1_QUICK.value,
            melee.Action.EDGE_JUMP_1_SLOW.value,
            melee.Action.EDGE_JUMP_2_QUICK.value,
            melee.Action.EDGE_JUMP_2_SLOW.value,
            melee.Action.ROLL_BACKWARD.value,
            melee.Action.ROLL_FORWARD.value,
            melee.Action.ENTRY.value,
            melee.Action.ENTRY_START.value,
            melee.Action.ENTRY_END.value,
            melee.Action.SLIDING_OFF_EDGE.value,
            melee.Action.DEAD_FALL.value,
            melee.Action.DEAD_DOWN.value,
            melee.Action.DEAD_LEFT.value,
            melee.Action.DEAD_RIGHT.value,
            melee.Action.DEAD_UP.value,
            melee.Action.DEAD_FLY_STAR.value,
            melee.Action.DEAD_FLY_STAR_ICE.value,
            melee.Action.DEAD_FLY.value,
            melee.Action.DEAD_FLY_SPLATTER.value,
            melee.Action.DEAD_FLY_SPLATTER_FLAT.value,
            melee.Action.DEAD_FLY_SPLATTER_ICE.value,
            melee.Action.DEAD_FLY_SPLATTER_FLAT_ICE.value,
            melee.Action.SPOTDODGE.value,
            melee.Action.TECH_MISS_DOWN.value,
            melee.Action.TECH_MISS_UP.value,
            melee.Action.EDGE_TEETERING.value,
            melee.Action.EDGE_TEETERING_START.value,
            melee.Action.EDGE_HANGING.value,
            melee.Action.EDGE_ROLL_SLOW.value,
            melee.Action.EDGE_ROLL_QUICK.value,
            melee.Action.EDGE_CATCHING.value,
            melee.Action.GRAB_PULL.value,
            melee.Action.GRAB_FOOT.value,
            melee.Action.GRAB_NECK.value,
            melee.Action.GROUND_ROLL_BACKWARD_DOWN.value,
            melee.Action.GROUND_ROLL_BACKWARD_UP.value,
            melee.Action.GROUND_ROLL_FORWARD_DOWN.value,
            melee.Action.GROUND_ROLL_FORWARD_UP.value,
            melee.Action.EDGE_ATTACK_SLOW.value,
            melee.Action.EDGE_ATTACK_QUICK.value,
            melee.Action.PUMMELED_HIGH.value,
            melee.Action.JUMPING_ARIAL_BACKWARD.value,
            melee.Action.JUMPING_ARIAL_FORWARD.value,
            melee.Action.GRAB_PULLING_HIGH.value
        ]

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError(
            "%s is an invalid controller port. \
                                         Must be 1, 2, 3, or 4."
            % value
        )
    return ivalue


class MeleeArgs:
    def __init__(
        self,
        mode,
        num_instances,
        port,
        opponent,
        debug,
        address,
        dolphin_port,
        dolphin_executable_path,
        connect_code,
        iso,
        best_sample_size,
    ):
        self.mode = mode
        self.num_instances = num_instances
        self.port = port
        self.opponent = opponent
        self.debug = debug
        self.address = address
        self.dolphin_port = dolphin_port
        self.dolphin_executable_path = dolphin_executable_path
        self.connect_code = connect_code
        self.iso = iso
        self.best_sample_size = best_sample_size


def parseArgs():
    parser = argparse.ArgumentParser(description="Melee")
    parser.add_argument(
        "--sample_size",
        "-s",
        type=int,
        help="The number of best samples to use to use for stream",
        default=20,
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="The mode to run the simulation in. Options are train or stream",
        default="train",
    )
    parser.add_argument(
        "--num_instances",
        "-n",
        type=int,
        help="The number of instances to run",
        default=1,
    )
    parser.add_argument(
        "--port",
        "-p",
        type=check_port,
        help="The controller port (1-4) your AI will play on",
        default=2,
    )
    parser.add_argument(
        "--opponent",
        "-o",
        type=check_port,
        help="The controller port (1-4) the opponent will play on",
        default=1,
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Debug mode. Creates a CSV of all game states",
    )
    parser.add_argument(
        "--address", "-a", default="127.0.0.1", help="IP address of Slippi/Wii"
    )
    parser.add_argument(
        "--dolphin_port", "-b", default=51441, type=int, help="IP address of Slippi/Wii"
    )
    parser.add_argument(
        "--dolphin_executable_path",
        "-e",
        default=None,
        help="The directory where dolphin is",
    )
    parser.add_argument(
        "--connect_code",
        "-t",
        default="",
        help="Direct connect code to connect to in Slippi Online",
    )
    parser.add_argument("--iso", default=None, type=str, help="Path to melee iso.")
    args = parser.parse_args()
    return MeleeArgs(
        mode=args.mode,
        num_instances=args.num_instances,
        port=args.port,
        opponent=args.opponent,
        debug=args.debug,
        address=args.address,
        dolphin_port=args.dolphin_port,
        dolphin_executable_path=args.dolphin_executable_path,
        connect_code=args.connect_code,
        iso=args.iso,
        best_sample_size=args.sample_size,
    )


def createMelee(args: MeleeArgs, instance_num: int):
    if args.mode == "stream":
        console = Console(
            path=args.dolphin_executable_path,
            slippi_port=args.dolphin_port + instance_num,
            blocking_input=False,
            polling_mode=False,
            save_replays=False,
        )
    else:
        console = Console(
            path=args.dolphin_executable_path,
            slippi_port=args.dolphin_port + instance_num,
            blocking_input=True,
            polling_mode=False,
            setup_gecko_codes=True,
            gfx_backend="Null",
            use_exi_inputs=False,
            enable_ffw=False,
            save_replays=False,
        )
    controller = Controller(
        console=console, port=args.port, type=melee.ControllerType.STANDARD
    )

    controller_opponent = Controller(
        console=console, port=args.opponent, type=melee.ControllerType.STANDARD
    )
    return MeleeCore(
        args=args,
        console=console,
        controller=controller,
        controller_opponent=controller_opponent,
    )


class MeleeCore:
    def __init__(
        self,
        args: MeleeArgs,
        console: Console,
        controller: Controller,
        controller_opponent: Controller,
    ):
        self.args = args
        self.console = console
        self.controller = controller
        self.controller_opponent = controller_opponent

    def run(self):
        self.console.run(iso_path=self.args.iso)

    def connect(self):
        if self.console.connect():
            print("Connected to console")
        else:
            print("Failed to connect to console")
        if self.controller.connect():
            print("Connected to port " + str(self.args.port))
        if self.controller_opponent.connect():
            print("Connected to port " + str(self.args.opponent))

    def stop(self):
        self.console.stop()
        print("=>>>>>>>>>>>>>>>>SHUTTING DOWN<<<<<<<<<<<<<<<<<=")

    def next_step(self):
        return self.console.step()


class MeleeSignalHandler:
    def __init__(self, melee: MeleeCore):
        self.melee = melee

    def signal_handler(self, signal, frame):
        self.melee.stop()
        sys.exit(0)


class AgentType(Enum):
    HUMAN = 0
    AI = 1
    CPU = 2


class AgentConfiguration:
    def __init__(
        self,
        character: melee.Character,
        type: AgentType,
        cpu_level: int,
        port: int,
    ):
        self.character = character
        self.type = type
        self.cpu_level = cpu_level
        self.port = port


class Agent:
    def __init__(
        self,
        melee: MeleeCore,
        agent_configuration: AgentConfiguration,
        controller: Controller,
        task_network: Optional[TaskNetwork2],
    ):
        self.melee = melee
        self.agent_configuration = agent_configuration
        self.controller = controller
        self.task_network = task_network
        self.input_count = 0
        self.prev_input = (0.5, 0.5, 0.5, 0.5, 0.0, False, False, False, False)

    def player(self, game_state: GameState) -> PlayerState:
        return game_state.players[self.agent_configuration.port]

    def create_action(self, input: Tensor):
        if self.task_network is None:
            return None
        # print(input.shape)
        output = self.task_network.forward(input)
        # print(output.shape)
        return output


class AgentScore:
    def __init__(
        self,
        agent: Agent,
        kill_count: int,
        death_count: int,
        damage_dealt: float,
        damage_received: float,
        center_advantage: float,
    ):
        self.agent = agent
        self.kill_count = kill_count
        self.death_count = death_count
        self.damage_dealt = damage_dealt
        self.damage_received = damage_received
        self.center_advantage = center_advantage
        self.unique_actions = set()


class AgentScoreDelta:
    def __init__(
        self,
        agent: Agent,
        kill_count_delta: int,
        death_count_delta: int,
        damage_dealt_delta: float,
        damage_received_delta: float,
        center_advantage_delta: float,
    ):
        self.agent = agent
        self.kill_count_delta = kill_count_delta
        self.death_count_delta = death_count_delta
        self.damage_dealt_delta = damage_dealt_delta
        self.damage_received_delta = damage_received_delta
        self.center_advantage_delta = center_advantage_delta
        self.action = 0


class GameStateDeltaProcessor:
    def __init__(self):

        self.game_state = None
        self.prev_game_state = None

    def process_state(self, game_state: GameState, agents: List[Tuple[Agent, Agent]]):
        self.prev_game_state = self.game_state
        self.game_state = game_state
        if self.prev_game_state is None:
            return
        delta_scores: List[AgentScoreDelta] = []
        for [agent1, agent2] in agents:
            agent1_score_delta = AgentScoreDelta(agent1, 0, 0, 0.0, 0.0, 0.0)
            player1 = agent1.player(self.game_state)
            prev_player1 = agent1.player(self.prev_game_state)
            player2 = agent2.player(self.game_state)
            prev_player2 = agent2.player(self.prev_game_state)
            player1_percent_change = player1.percent - prev_player1.percent
            player1_stock_change = int(player1.stock) - int(prev_player1.stock)
            player2_percent_change = player2.percent - prev_player2.percent
            player2_stock_change = int(player2.stock) - int(prev_player2.stock)
            # print(f"player1_stock_change: {player1_stock_change}")
            # print(f"player2_stock_change: {player2_stock_change}")
            if abs(player1.x) < 25:
                agent1_score_delta.center_advantage_delta = 1
            if player1_stock_change < 0:
                # lose stocks
                agent1_score_delta.death_count_delta = 1
            if player1_percent_change > 0:
                # gain stocks
                agent1_score_delta.damage_received_delta = player1_percent_change
            if player2_stock_change < 0:
                # implement tests to see if player1 actually killed player2
                agent1_score_delta.kill_count_delta = 1
            if player2_percent_change > 0:
                # implement tests to see if player1 is actually damaging player2
                agent1_score_delta.damage_dealt_delta = player2_percent_change
            agent1_score_delta.action = player1.action.value

            delta_scores.append(agent1_score_delta)
        return delta_scores


class GameStateEvaluator:
    def __init__(
        self,
        melee: MeleeCore,
        agents: List[Agent],
        game_state_processor: GameStateDeltaProcessor,
    ):
        self.melee = melee
        self.agentPortMap = {agent.agent_configuration.port: agent for agent in agents}
        self.agents = agents
        self.game_state_processor = game_state_processor
        self.agent_scores = {
            agent.agent_configuration.port: AgentScore(agent, 0, 0, 0.0, 0.0, 0.0)
            for agent in agents
        }
        self.excluded_actions = excluded_actions()

    def score_agents(self, game_state: GameState):
        delta_scores = self.game_state_processor.process_state(
            game_state,
            [(self.agents[0], self.agents[1]), (self.agents[1], self.agents[0])],
        )
        if delta_scores is None:
            return self.agent_scores
        for delta_score in delta_scores:
            self.agent_scores[
                delta_score.agent.agent_configuration.port
            ].kill_count += delta_score.kill_count_delta
            self.agent_scores[
                delta_score.agent.agent_configuration.port
            ].death_count += delta_score.death_count_delta
            self.agent_scores[
                delta_score.agent.agent_configuration.port
            ].damage_dealt += delta_score.damage_dealt_delta
            self.agent_scores[
                delta_score.agent.agent_configuration.port
            ].damage_received += delta_score.damage_received_delta
            if delta_score.action not in self.excluded_actions:
                self.agent_scores[
                    delta_score.agent.agent_configuration.port
                ].center_advantage += delta_score.center_advantage_delta
            if delta_score.action not in self.excluded_actions:
                self.agent_scores[
                    delta_score.agent.agent_configuration.port
                ].unique_actions.add(delta_score.action)
                # if delta_score.agent.agent_configuration.port == 1:
                #     print(f"agent {delta_score.agent.agent_configuration.port} action: {delta_score.action} all actions: {self.agent_scores[delta_score.agent.agent_configuration.port].unique_actions}")

        return self.agent_scores


class MeleeConfiguration:
    def __init__(self, agents: List[AgentConfiguration], stage: melee.Stage):
        self.agents = agents
        self.stage = stage


from enum import Enum


def stageToString(stage: melee.Stage):
    if stage == melee.Stage.FINAL_DESTINATION:
        return "FINALDESTINATION"
    elif stage == melee.Stage.BATTLEFIELD:
        return "BATTLEFIELD"
    elif stage == melee.Stage.POKEMON_STADIUM:
        return "POKEMONSTADIUM"
    elif stage == melee.Stage.DREAMLAND:
        return "DREAMLAND"
    elif stage == melee.Stage.FOUNTAIN_OF_DREAMS:
        return "FOUNTAINOFDREAMS"
    elif stage == melee.Stage.YOSHIS_STORY:
        return "YOSHISSTORY"


def characterToString(character: melee.Character):
    if character == melee.Character.DOC:
        return "DOC"
    elif character == melee.Character.DK:
        return "DK"
    elif character == melee.Character.CPTFALCON:
        return "CPTFALCON"
    elif character == melee.Character.GANONDORF:
        return "GANONDORF"
    elif character == melee.Character.FOX:
        return "FOX"
    elif character == melee.Character.FALCO:
        return "FALCO"
    elif character == melee.Character.LINK:
        return "LINK"
    elif character == melee.Character.MARIO:
        return "MARIO"
    elif character == melee.Character.YOSHI:
        return "YOSHI"
    elif character == melee.Character.LUIGI:
        return "LUIGI"
    elif character == melee.Character.PIKACHU:
        return "PIKACHU"
    elif character == melee.Character.KIRBY:
        return "KIRBY"
    elif character == melee.Character.BOWSER:
        return "BOWSER"
    elif character == melee.Character.SHEIK:
        return "SHEIK"
    elif character == melee.Character.NESS:
        return "NESS"
    elif character == melee.Character.PEACH:
        return "PEACH"
    elif character == melee.Character.POPO:
        return "POPO"
    elif character == melee.Character.NANA:
        return "NANA"
    elif character == melee.Character.SAMUS:
        return "SAMUS"
    elif character == melee.Character.JIGGLYPUFF:
        return "JIGGLYPUFF"
    elif character == melee.Character.MEWTWO:
        return "MEWTWO"
    elif character == melee.Character.MARTH:
        return "MARTH"
    elif character == melee.Character.ZELDA:
        return "ZELDA"
    elif character == melee.Character.YLINK:
        return "YLINK"
    elif character == melee.Character.PICHU:
        return "PICHU"
    elif character == melee.Character.GAMEANDWATCH:
        return "GAMEANDWATCH"
    elif character == melee.Character.ROY:
        return "ROY"


class SimulationState(Enum):
    RUNNING = 0
    GAME_OVER = 1
    MENU = 2
    SUDDEN_DEATH = 3
    GAME_ACTIVE_SIMULATION_END = 4


class ActionTracker:
    def __init__(self, unique_size: int):
        self.unique_size = unique_size
        self.unique_action_set = deque(maxlen=unique_size)
        self.actions = []
        self.excluded_actions = excluded_actions()

    def add_action(self, action: int):
        if action not in self.excluded_actions:
            is_unique = action not in self.unique_action_set

            # if len(self.unique_action_set) > self.unique_size:
            #     self.unique_action_set.
            if is_unique:
                self.unique_action_set.append(action)
                self.actions.append(action)
                # print(len(self.actions))

    def get_actions(self):

        return self.actions


class MeleeSimulation:

    def __init__(
        self,
        melee: MeleeCore,
        melee_config: MeleeConfiguration,
        use_action_coords: bool,
    ):
        self.melee = melee
        self.melee_config = melee_config
        self.controller_helper = ControllerHelper()
        self.use_action_coords = use_action_coords
        self.action_tracker = [ActionTracker(5), ActionTracker(5)]

    def set_config(self, melee_config: MeleeConfiguration):
        self.melee_config = melee_config

    def simulation_step(
        self,
        game_state: GameState,
        game_state_evaluator: GameStateEvaluator,
        menu_helper: melee.menuhelper.MenuHelper,
        trainMode : bool
    ):
        if game_state.menu_state in [
            melee.enums.Menu.IN_GAME,
            melee.enums.Menu.SUDDEN_DEATH,
        ]:
            agent_scores = game_state_evaluator.score_agents(game_state)
            game_over = self.isGameOver(game_state, agent_scores, trainMode)
            if game_over[0]:
                return (
                    game_state_evaluator.agent_scores,
                    game_over[2],
                )
            else:
                self.handle_game_step(game_state, game_state_evaluator.agents)
                return (game_state_evaluator.agent_scores, SimulationState.RUNNING)
        elif game_state.menu_state == melee.enums.Menu.MAIN_MENU:
            menu_helper.choose_versus_mode(
                game_state, game_state_evaluator.agents[0].controller
            )
        else:
            # menu_helper.choose_character(
            #         gamestate=game_state,
            #         character=melee.enums.Character.FOX,
            #         controller=game_state_evaluator.agents[0].controller,
            #         cpu_level=3,
            #     )

            all_characters_selected = False
            all_cpu_level_selected = False
            for i, agent_config in enumerate(self.melee_config.agents):
                # print(agent_config.character)
                # print(melee.enums.Character.FOX)
                choose_character(
                    gamestate=game_state,
                    character=agent_config.character,
                    controller=game_state_evaluator.agents[i].controller,
                    cpu_level=agent_config.cpu_level,
                )
            if game_state.players:
                all_characters_selected = all(
                    game_state_evaluator.agents[i].player(game_state).character
                    == self.melee_config.agents[i].character
                    for i in range(len(self.melee_config.agents))
                )
                all_cpu_level_selected = all(
                    game_state_evaluator.agents[i].player(game_state).cpu_level
                    == self.melee_config.agents[i].cpu_level
                    for i in range(len(self.melee_config.agents))
                )
                # print(all_characters_selected)
                # print(all_cpu_level_selected)
                # print(
                #     game_state_evaluator.agents[0].player(game_state).cpu_level
                #     == self.melee_config.agents[0].cpu_level
                # )
                # print(
                #     game_state_evaluator.agents[1].player(game_state).cpu_level
                #     == self.melee_config.agents[1].cpu_level
                # )
            if (
                all_characters_selected
                and all_cpu_level_selected
                and game_state.menu_state == melee.enums.Menu.CHARACTER_SELECT
            ):
                # print("PRESS START")
                if game_state.frame % 2 == 0:
                    game_state_evaluator.agents[0].controller.press_button(
                        melee.enums.Button.BUTTON_START
                    )
                    game_state_evaluator.agents[1].controller.press_button(
                        melee.enums.Button.BUTTON_START
                    )
                else:
                    game_state_evaluator.agents[0].controller.release_button(
                        melee.enums.Button.BUTTON_START
                    )
                    game_state_evaluator.agents[1].controller.release_button(
                        melee.enums.Button.BUTTON_START
                    )
            elif game_state.menu_state == melee.enums.Menu.STAGE_SELECT:
                menu_helper.choose_stage(
                    self.melee_config.stage,
                    game_state,
                    game_state_evaluator.agents[0].controller,
                )

        return (game_state_evaluator.agent_scores, SimulationState.MENU)

    def isGameOver(self, game_state: GameState, agent_scores: Dict[int, AgentScore], trainMode : bool):
        players: List[Tuple[PlayerState, int]] = list(
            map(lambda x: (agent_scores[x].agent.player(game_state), x), agent_scores)
        )
        
        if game_state.frame / (60 * 60) >= 8:
            return (True, -1, SimulationState.SUDDEN_DEATH)
        if agent_scores[1].agent.player(game_state).stock < 4 and trainMode:
            return (True, agent_scores[1].agent.controller.port, SimulationState.GAME_ACTIVE_SIMULATION_END)
        for player, index in players:
            if player.stock <= 0:
                return (True, index, SimulationState.GAME_OVER)
        return (False, -1, SimulationState.RUNNING)

    def handle_game_step(self, game_state: GameState, agents: List[Agent]):
        task_input: Tensor
        # print(agents[0].agent_configuration.type)
        # print(agents[0].agent_configuration.type == AgentType.AI)
        # print(agents[0].agent_configuration.type is AgentType.CPU)
        # print(agents[0].agent_configuration.type == "AI")
        for i, agent in enumerate(filter(lambda a: a.agent_configuration.type == "AI", agents)):
            if self.use_action_coords:
                input_tensor, input_tensor_2, controller_tensor, input_action_tensor, input_action_tensor_2 = game_state_to_tensor(
                    game_state,
                    agent.player(game_state),
                    (
                        agents[1].player(game_state)
                        if agent == agents[0]
                        else agents[0].player(game_state)
                    ),
                )
                task_input = torch.cat(
                    (input_tensor.flatten(), input_tensor_2.flatten(),  input_action_tensor.flatten(), input_action_tensor_2.flatten(), controller_tensor.flatten())
                )
                # if game_state.frame % (60 * 5) == 0:
                #     print(input_tensor.flatten())
                #     print(input_tensor_2.flatten())
                #     non_zero_indices = torch.nonzero(input_action_tensor.flatten(), as_tuple=True)
                #     print(non_zero_indices)
                #     non_zero_indices_2 = torch.nonzero(input_action_tensor_2.flatten(), as_tuple=True)
                #     print(non_zero_indices_2)
            else:
                input_tensor, controller_tensor = game_state_to_tensor_action_normalized(
                    game_state,
                    agent.player(game_state),
                    (
                        agents[1].player(game_state)
                        if agent == agents[0]
                        else agents[0].player(game_state)
                    ),
                )
                task_input = torch.cat((input_tensor.flatten(), controller_tensor.flatten()))
            output = agent.create_action(task_input)
            if output is not None:
                buttons, analog, c_analog = output_tensor_to_controller_tensors(
                    output  # torch.sigmoid(output)
                )
                buttons = torch.sigmoid(buttons)
                threshold = 0.5
                press_a: bool = buttons[0, 0] > threshold
                press_b: bool = buttons[0, 1] > threshold
                press_y: bool = buttons[0, 2] > threshold
                press_z: bool = buttons[0, 3] > threshold
                left_shoulder = (
                    (buttons[0, 4].item() - threshold) / (1 - threshold)
                    if buttons[0, 4].item() > threshold
                    else 0.0
                )
                main_stick_x, main_stick_y = self.controller_helper.processAnalog(
                    analog
                )
                c_analog_x, c_analog_y = self.controller_helper.processAnalog(c_analog)
                self.controller_helper.processMessage(
                    {
                        "a": press_a,
                        "b": press_b,
                        "y": press_y,
                        "z": press_z,
                        "leftShoulder": left_shoulder,
                        "mainStickX": main_stick_x,
                        "mainStickY": main_stick_y,
                        "cStickX": c_analog_x,
                        "cStickY": c_analog_y,
                    },
                    agent.controller,
                )
                new_input = (
                    main_stick_x,
                    main_stick_y,
                    c_analog_x,
                    c_analog_y,
                    left_shoulder,
                    press_a,
                    press_b,
                    press_y,
                    press_z,
                )
                input_delta = 0
                input_delta = sum(
                    1 for new, prev in zip(new_input, agent.prev_input) if new != prev
                )
                agent.prev_input = new_input
                agent.input_count += input_delta
                self.action_tracker[i].add_action(agent.player(game_state).action.value)

def simulation(
    queue: Queue,
    score_queue: Queue,
    args: MeleeArgs,
    use_action_coords: bool,
    instance_num: int,
):
    meleeCore = createMelee(args, instance_num)
    meleeCore.run()
    signal_handler = MeleeSignalHandler(meleeCore)
    signal.signal(signal.SIGINT, signal_handler.signal_handler)
    print("Connecting...")
    # time.sleep(5)
    print("Connecting!")
    meleeCore.connect()

    menu_helper = melee.menuhelper.MenuHelper
    print("simulation starting...")
    id: str
    network: TaskNetwork2
    agent_config: AgentConfiguration
    cpu_config: AgentConfiguration
    stage: melee.Stage
    # fetch from queue
    while True:
        # print("get next network")
        (id, network, network2, agent_config, cpu_config, stage) = queue.get()
        agent_configuration_list = [
            agent_config,
            cpu_config,
        ]
        print("id", id)
        
        agent_score: AgentScore
        first_step = True
        state = SimulationState.MENU
        print(f"start loop {instance_num}")
        if (
            agent_config.character == melee.Character.CPTFALCON
            and cpu_config.character == melee.Character.CPTFALCON
        ):
            continue
        total_games = 1 if args.mode == "train" else 1
        total_scores = {
            "kill_count": 0,
            "death_count": 0,
            "damage_dealt": 0.0,
            "damage_received": 0.0,
            "center_advantage": 0.0,
            "unique_action_count": 0,
            "total_frames": 0,
            "input_count": 0,
            "rolling_action_count": 0,
        }

        for game in range(total_games):
            melee_config = MeleeConfiguration(agent_configuration_list, stage)
            meleeSimulation = MeleeSimulation(meleeCore, melee_config, use_action_coords)
            agents = [
                Agent(meleeCore, agent_config, meleeCore.controller, network),
                Agent(meleeCore, cpu_config, meleeCore.controller_opponent, network2),
            ]
            game_state_evaluator = GameStateEvaluator(
                meleeCore,
                agents,
                GameStateDeltaProcessor(),
            )
            while True:
                game_state = meleeCore.next_step()
                if first_step:
                    print(
                        f"({instance_num}) agent {agent_config.character} vs cpu {cpu_config.character} {cpu_config.cpu_level}"
                    )
                    first_step = False
                # if (
                #      args.mode != "train"
                # ):  # agent_config.character == melee.Character.CPTFALCON and cpu_config.character == melee.Character.CPTFALCON:
                #     if (
                #         game_state.frame % (60 * 1) == 0
                #         and game_state.menu_state == melee.Menu.IN_GAME
                #     ):
                #         print(
                #             f"({instance_num})  agent {agent_config.character} vs cpu {cpu_config.character} frame: {game_state.frame} agent x: {agents[0].player(game_state).x} cpu x: {agents[1].player(game_state).x} agent percent: {agents[0].player(game_state).percent}"
                #         )
                # print(game_state)
                # print(agent_config.character)
                # print(cpu_config.character)
                # print(stage)
                if game_state is None:
                    print("Game state is None")
                    meleeCore.stop()
                    # simulation(queue, score_queue, args, use_action_coords, instance_num)

                    break
                (score, state) = meleeSimulation.simulation_step(
                    game_state, game_state_evaluator, menu_helper, args.mode == "train"
                )
                ai_port = meleeCore.controller.port
                agent_score = score[ai_port]
                if (
                    state == SimulationState.GAME_OVER
                    or state == SimulationState.SUDDEN_DEATH
                ):
                    # print("Game Over")
                    meleeCore.controller.release_all()
                    meleeCore.controller_opponent.release_all()
                    break
                if state == SimulationState.GAME_ACTIVE_SIMULATION_END:
                    meleeCore.controller.release_all()
                    meleeCore.controller_opponent.release_all()
                    break
            # print((id, agent_score.kill_count, agent_score.death_count, agent_score.damage_dealt, agent_score.damage_received))
            # print(meleeSimulation.action_tracker[0].actions)
            
                
                

            if state == SimulationState.GAME_OVER or state == SimulationState.GAME_ACTIVE_SIMULATION_END:
                total_scores["kill_count"] += agent_score.kill_count
                total_scores["death_count"] += agent_score.death_count
                total_scores["damage_dealt"] += agent_score.damage_dealt
                total_scores["damage_received"] += agent_score.damage_received
                total_scores["center_advantage"] += agent_score.center_advantage
                total_scores["unique_action_count"] += len(agent_score.unique_actions)
                total_scores["total_frames"] += int(game_state.frame)
                total_scores["input_count"] += agents[0].input_count
                total_scores["rolling_action_count"] += len(meleeSimulation.action_tracker[0].actions)

            if state == SimulationState.GAME_ACTIVE_SIMULATION_END and args.mode == "train":
                while(game_state.menu_state == melee.Menu.IN_GAME):
                    game_state = meleeCore.next_step()
                    meleeCore.controller.tilt_analog(melee.Button.BUTTON_MAIN,0, .5)
                    if game_state.frame % 2 == 0:
                        meleeCore.controller.press_button(melee.Button.BUTTON_X)
                    else:
                        meleeCore.controller.release_button(melee.Button.BUTTON_X)

        average_scores = {key: value / total_games for key, value in total_scores.items()}
        print(f"average_scores: {average_scores}")
        print(f"total_scores: {total_scores}")
        score_dict = {
            "id": id,
            "kill_count": average_scores["kill_count"],
            "death_count": average_scores["death_count"],
            "damage_dealt": average_scores["damage_dealt"],  # agents[1].player(game_state).percent,
            "damage_received": average_scores["damage_received"],  # agents[0].player(game_state).percent,
            "center_advantage": average_scores["center_advantage"],
            "unique_action_count": average_scores["unique_action_count"],
            "total_frames": average_scores["total_frames"],
            "input_count": average_scores["input_count"],
            "rolling_action_count": average_scores["rolling_action_count"],
            "cpu_level": cpu_config.cpu_level,
            "stage": stageToString(melee_config.stage),
            "character": characterToString(agent_config.character),
            "opponent_character": characterToString(cpu_config.character),
            "l2_norm": network.calculate_l2_norm(),
            "l1_norm": network.calculate_l1_norm(),
        }
        # print(f"{score_dict}")
        # score_queue.put(score_dict)
        # print("last before send")
        if args.mode == "train":
            score_queue.put(score_dict)
        # print("last after send")

        # else:
        #     score_queue.put((id, agent_score))
    # simulation.simulation_step(game_state, game_state_evaluator, menu_helper)


class Task:
    def __init__(
        self,
        id: str,
        agent_config: AgentConfiguration,
        network_genome: Optional[NetworkGenome],
    ):
        self.id = id
        self.agent_config = agent_config
        self.network_genome = network_genome


class AgentVsCPUTask:
    def __init__(self, id: str, agent_task: Task, cpu_task: Task, stage: melee.Stage):
        self.id = id
        self.agent_task = agent_task
        self.cpu_task = cpu_task
        self.stage = stage


def parseCharacter(character_string: str):

    character_table = {
        "mario": melee.Character.MARIO,
        "fox": melee.Character.FOX,
        "falco": melee.Character.FALCO,
        "cptfalcon": melee.Character.CPTFALCON,
        "marth": melee.Character.MARTH,
        "pikachu": melee.Character.PIKACHU,
        "link": melee.Character.LINK,
        "yoshi": melee.Character.YOSHI,
        "doc": melee.Character.DOC,
        "dk": melee.Character.DK,
        "bowser": melee.Character.BOWSER,
        "gameandwatch": melee.Character.GAMEANDWATCH,
        "ganondorf": melee.Character.GANONDORF,
        "jigglypuff": melee.Character.JIGGLYPUFF,
        "kirby": melee.Character.KIRBY,
        "luigi": melee.Character.LUIGI,
        "mewtwo": melee.Character.MEWTWO,
        "ness": melee.Character.NESS,
        "peach": melee.Character.PEACH,
        "pichu": melee.Character.PICHU,
        "roy": melee.Character.ROY,
        "sheik": melee.Character.SHEIK,
        "ylink": melee.Character.YLINK,
        "zelda": melee.Character.ZELDA,
        "popo": melee.Character.POPO,
        "samus": melee.Character.SAMUS,
    }
    return character_table[character_string.lower()]


def parseStage(stage_string: str):
    stage_table = {
        "finaldestination": melee.Stage.FINAL_DESTINATION,
        "battlefield": melee.Stage.BATTLEFIELD,
        "dreamland": melee.Stage.DREAMLAND,
        "pokemonstadium": melee.Stage.POKEMON_STADIUM,
    }
    return stage_table[stage_string.lower()]


def dict_to_classes(data: Dict[str, Any]) -> AgentVsCPUTask:
    def create_task(task_data: Dict[str, Any]) -> Task:
        agent_config_data = task_data["agentConfig"]
        agent_config = AgentConfiguration(
            character=parseCharacter(agent_config_data["character"]),
            type=agent_config_data["type"],
            cpu_level=agent_config_data["cpuLevel"],
            port=agent_config_data["port"],
        )
        network_genome = (
            json_to_network_genome(task_data["networkGenome"])
            if task_data.get("networkGenome")
            else None
        )
        return Task(
            id=task_data["id"], agent_config=agent_config, network_genome=network_genome
        )

    agent_task = create_task(data["agentTask"])
    cpu_task = create_task(data["cpuTask"])
    return AgentVsCPUTask(
        id=data["id"],
        agent_task=agent_task,
        cpu_task=cpu_task,
        stage=parseStage(data["stage"]),
    )


def fetch_network_genome(api_url, queue: Queue, substrate: Substrate):

    while True:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = json.loads(response.text)

            task = dict_to_classes(data)
            network_builder = NetworkBuilder(DefaultActivationFunctionMapper())
            network_processor_factory = NetworkProcessorFactory(
                network_builder, False, 1, 0.01
            )
            network_processor = network_processor_factory.createProcessor(
                task.agent_task.network_genome
            )
            cppn_query_instance = CPPNConnectionQuery(network_processor, 1.0, 0.3)
            network = TaskNetwork2(substrate, cppn_query_instance)
            network2 = None
            if task.cpu_task.network_genome is not None:
                network_processor2 = network_processor_factory.createProcessor(
                    task.cpu_task.network_genome
                )
                cppn_query_instance2 = CPPNConnectionQuery(network_processor2, 1.0, 0.3)
                network2 = TaskNetwork2(substrate, cppn_query_instance2)
            queue.put(
                [
                    data["id"],
                    network,
                    network2,
                    task.agent_task.agent_config,
                    task.cpu_task.agent_config,
                    task.stage,
                ]
            )

        else:
            print(
                f"Failed to fetch network genome from {api_url}. Status code: {response.status_code}"
            )
            time.sleep(1)


def score_queue_process(score_queue: Queue):
    id: str
    score: AgentScore
    while True:
        score = score_queue.get()
        id = score["id"]
        # print(score)
        requests.post(
            "http://192.168.0.100:8080/score",
            json={
                "id": id,
                "stocksTaken": score["kill_count"],
                "stocksLost": score["death_count"],
                "damageDone": score["damage_dealt"],
                "damageTaken": score["damage_received"],
                "centerAdvantage": score["center_advantage"],
                "uniqueActionCount": score["unique_action_count"],
                "totalInputs": score["input_count"],
                "rollingActionCount": score["rolling_action_count"],
                "totalFrames": score["total_frames"],
                "cpuLevel": score["cpu_level"],
                "stage": score["stage"],
                "character": score["character"],
                "opponentCharacter": score["opponent_character"],
                "l2Norm": score["l2_norm"],
                "l1Norm": score["l1_norm"],
            },
        )
        # print("send request!")


def stageToInt(stage: melee.Stage):
    if stage == melee.Stage.FINAL_DESTINATION:
        return 0
    elif stage == melee.Stage.BATTLEFIELD:
        return 1
    elif stage == melee.Stage.DREAMLAND:
        return 2
    elif stage == melee.Stage.POKEMON_STADIUM:
        return 3
    elif stage == melee.Stage.FOUNTAIN_OF_DREAMS:
        return 4
    elif stage == melee.Stage.YOSHIS_STORY:
        return 5


def characterToInt(character: melee.Character):
    if character == melee.Character.MARIO:
        return 0
    elif character == melee.Character.FOX:
        return 1
    elif character == melee.Character.CPTFALCON:
        return 2
    elif character == melee.Character.DK:
        return 3
    elif character == melee.Character.KIRBY:
        return 4
    elif character == melee.Character.BOWSER:
        return 5
    elif character == melee.Character.LINK:
        return 6
    elif character == melee.Character.SHEIK:
        return 7
    elif character == melee.Character.NESS:
        return 8
    elif character == melee.Character.PEACH:
        return 9
    elif character == melee.Character.POPO:
        return 10
    elif character == melee.Character.NANA:
        return 11
    elif character == melee.Character.PIKACHU:
        return 12
    elif character == melee.Character.SAMUS:
        return 13
    elif character == melee.Character.YOSHI:
        return 14
    elif character == melee.Character.JIGGLYPUFF:
        return 15
    elif character == melee.Character.MEWTWO:
        return 16
    elif character == melee.Character.LUIGI:
        return 17
    elif character == melee.Character.MARTH:
        return 18
    elif character == melee.Character.ZELDA:
        return 19
    elif character == melee.Character.YLINK:
        return 20
    elif character == melee.Character.DOC:
        return 21
    elif character == melee.Character.FALCO:
        return 22
    elif character == melee.Character.PICHU:
        return 23
    elif character == melee.Character.GAMEANDWATCH:
        return 24
    elif character == melee.Character.GANONDORF:
        return 25
    elif character == melee.Character.ROY:
        return 26


# -246, 246, 188, -140
def scale_to_custom_range(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val) * 2 - 1


def game_state_to_tensor(
    game_state: GameState, agent_player: PlayerState, opponent_player: PlayerState
):
    input_tensor = torch.zeros((2, 24))
    input_action_tensor = torch.zeros(26, 15)
    input_action_tensor_2 = torch.zeros(26, 15)
    controller_tensor = torch.zeros(1, 9)
    controller_tensor[0, 0] =1 if  agent_player.controller_state.button[melee.Button.BUTTON_A] else 0
    controller_tensor[0, 1] =1 if  agent_player.controller_state.button[melee.Button.BUTTON_B] else 0
    controller_tensor[0, 2] =1 if  agent_player.controller_state.button[melee.Button.BUTTON_Y] else 0
    controller_tensor[0, 3] =1 if  agent_player.controller_state.button[melee.Button.BUTTON_Z] else 0
    controller_tensor[0, 4] =1 if  agent_player.controller_state.l_shoulder else 0
    main_stick = agent_player.controller_state.main_stick
    c_stick = agent_player.controller_state.c_stick
    controller_tensor[0, 5] = main_stick[0]
    controller_tensor[0, 6] = main_stick[1]
    controller_tensor[0, 7] = c_stick[0]
    controller_tensor[0, 8] = c_stick[1]
    
    
    for i, player in enumerate([agent_player, opponent_player]):
        input_tensor[i, 0] = player.percent / 100
        input_tensor[i, 1] = player.stock / 4
        input_tensor[i, 2] = (
            player.x / 100
        )  # scale_to_custom_range(player.x, -246, 246)
        input_tensor[i, 3] = (
            player.y / 100
        )  # scale_to_custom_range(player.y, -140, 188)
        input_tensor[i, 4] = 1 if player.facing == True else 0
        input_tensor[i, 5] = player.jumps_left / 2.0
        input_tensor[i, 6] = player.shield_strength / 60.0
        input_tensor[i, 7] = stageToInt(game_state.stage) / 6.0
        input_tensor[i, 8] = characterToInt(player.character) / 26.0
        input_tensor[i, 9] = player.action_frame / 60
        input_tensor[i, 10] = 1 if player.on_ground else 0
        input_tensor[i, 11] = 1 if player.off_stage else 0
        input_tensor[i, 12] = melee.stages.EDGE_POSITION[game_state.stage] / 100.0
        input_tensor[i, 13] = -melee.stages.EDGE_POSITION[game_state.stage] / 100.0
        input_tensor[i, 14] = player.hitstun_frames_left / 60.0
        input_tensor[i, 15] = player.hitlag_left / 60.0
        input_tensor[i, 16] = player.speed_x_attack / 5
        input_tensor[i, 17] = player.speed_y_attack / 5
        input_tensor[i, 18] = player.speed_ground_x_self / 5
        input_tensor[i, 19] = player.speed_y_self / 5
        input_tensor[i, 20] = player.speed_air_x_self / 5
        input_tensor[i, 21] = 1 if player.moonwalkwarning else 0
        input_tensor[i, 22] = 1 if player.invulnerable else 0
        input_tensor[i, 23] = 1 if player.is_powershield else 0
        

    linear_index = min(385, agent_player.action.value)
    row = linear_index // 15
    col = linear_index % 15
    input_action_tensor[row, col] = 1
    linear_index = min(385, opponent_player.action.value)
    row = linear_index // 15
    col = linear_index % 15
    input_action_tensor_2[row, col] = 1
    return input_tensor[0, :].reshape(4, 6), input_tensor[1, :].reshape(4, 6), controller_tensor, input_action_tensor, input_action_tensor_2


def game_state_to_tensor_action_normalized(
    game_state: GameState, agent_player: PlayerState, opponent_player: PlayerState
):
    input_tensor = torch.zeros((2, 25))
    controller_tensor = torch.zeros(1, 9)
    controller_tensor[0, 0] =1 if  agent_player.controller_state.button[melee.Button.BUTTON_A] else 0
    controller_tensor[0, 1] =1 if  agent_player.controller_state.button[melee.Button.BUTTON_B] else 0
    controller_tensor[0, 2] =1 if  agent_player.controller_state.button[melee.Button.BUTTON_Y] else 0
    controller_tensor[0, 3] =1 if  agent_player.controller_state.button[melee.Button.BUTTON_Z] else 0
    controller_tensor[0, 4] =1 if  agent_player.controller_state.l_shoulder else 0
    main_stick = agent_player.controller_state.main_stick
    c_stick = agent_player.controller_state.c_stick
    controller_tensor[0, 5] = main_stick[0]
    controller_tensor[0, 6] = main_stick[1]
    controller_tensor[0, 7] = c_stick[0]
    controller_tensor[0, 8] = c_stick[1]
    # input_action_tensor = torch.zeros((2, 386))
    for i, player in enumerate([agent_player, opponent_player]):
        input_tensor[i, 0] = player.percent / 100
        input_tensor[i, 1] = player.stock / 4
        input_tensor[i, 2] = (
            player.x / 100
        )  # scale_to_custom_range(player.x, -246, 246)
        input_tensor[i, 3] = (
            player.y / 100
        )  # scale_to_custom_range(player.y, -140, 188)
        input_tensor[i, 4] = 1 if player.facing == True else -1
        input_tensor[i, 5] = player.jumps_left / 2
        input_tensor[i, 6] = player.shield_strength / 60
        input_tensor[i, 7] = min(385, player.action.value) / 385
        input_tensor[i, 8] = stageToInt(game_state.stage) / 6
        input_tensor[i, 9] = characterToInt(player.character) / 26
        input_tensor[i, 10] = player.action_frame / 120
        input_tensor[i, 11] = 1 if player.on_ground else 0
        input_tensor[i, 12] = 1 if player.off_stage else 0
        input_tensor[i, 13] = melee.stages.EDGE_POSITION[game_state.stage] / 100.0
        input_tensor[i, 14] = -melee.stages.EDGE_POSITION[game_state.stage] / 100.0
        input_tensor[i, 15] = player.hitstun_frames_left / 60.0
        input_tensor[i, 16] = player.hitlag_left / 60.0
        input_tensor[i, 17] = player.speed_x_attack / 5
        input_tensor[i, 18] = player.speed_y_attack / 5
        input_tensor[i, 19] = player.speed_ground_x_self / 5
        input_tensor[i, 20] = player.speed_y_self / 5
        input_tensor[i, 21] = player.speed_air_x_self / 5
        input_tensor[i, 22] = 1 if player.moonwalkwarning else 0
        input_tensor[i, 23] = 1 if player.invulnerable else 0
        input_tensor[i, 24] = 1 if player.is_powershield else 0

    return input_tensor, controller_tensor


def output_tensor_to_controller_tensors(output_tensor: Tensor):
    buttons = output_tensor[:, :5].reshape(1, 5)
    analog = output_tensor[:, 5:54].reshape(7, 7)
    c_analog = output_tensor[:, 54:103].reshape(7, 7)
    return buttons, analog, c_analog


def main():
    args = parseArgs()
    use_action_coords = True
    width = 6 if use_action_coords else 5
    height = 4 if use_action_coords else 5
    action_width = 15
    action_height = 26
    bias_coords = [(0, 0, -1.1)]
    input_coords = [
        (y, x, -1)
        for y in np.linspace(-1, 1, height)  # for z in np.linspace(-.1, .1, 3)
        for x in np.linspace(-1, 0, width)
    ]
    input_coords_2 = [
        (y, x, -1)
        for y in np.linspace(-1, 1, height)  # for z in np.linspace(-.1, .1, 3)
        for x in np.linspace(0, 1, width)
    ]
    controller_coords = [
        (x, x, -0.95)
        # for y in np.linspace(-1, 1, 1)
        for x in np.linspace(-1, 1, 9)
    ]
    action_coords = [
        (y, x, -0.9)
        for y in np.linspace(-1, 0, action_height)
        for x in np.linspace(-1, 0, action_width)
    ]
    action_coords_2 = [
        (y, x, -0.9)
        for y in np.linspace(0, 1, action_height)
        for x in np.linspace(0, 1, action_width)
    ]
    # for y in np.linspace(-1, 1, 1)
    hidden_coords = [
        [(y, x, 0) for y in np.linspace(-1, 1, 12) for x in np.linspace(-1, 1, 12)]
        # for z in np.linspace(-0.8, 0.8, round(5))
    ]
    output_width = 5
    output_height = 1
    analog_width = 7
    analog_height = 7
    analog_coords = [
        (y, x, 0.9)
        for y in np.linspace(-1, 1, analog_height)
        for x in np.linspace(-1, 0, analog_width)
    ]
    c_analog_coords = [
        (y, x, 0.9)
        for y in np.linspace(-1, 1, analog_height)
        for x in np.linspace(0, 1, analog_width)
    ]
    output_coords = [
        (y, x, 1)
        for y in np.linspace(-1, 1, output_height)
        for x in np.linspace(-1, 1, output_width)
    ]

    all_input_coords = (
        input_coords + input_coords_2  + (action_coords + action_coords_2) + controller_coords if use_action_coords else input_coords + input_coords_2 + controller_coords
    ) 
    all_output_coords = output_coords + analog_coords + c_analog_coords
    substrate = Substrate(
        all_input_coords, hidden_coords, all_output_coords, bias_coords
    )
    num_instances = args.num_instances
    mode = args.mode  # train or stream
    print(mode)
    print(num_instances)
    manager = Manager()
    queue = manager.Queue(num_instances *4 if args.mode == "train" else num_instances)
    score_queue = manager.Queue(num_instances * 5 if args.mode == "train" else num_instances)
    api_url = (
        "http://192.168.0.100:8080/networkGenome"
        if mode == "train"
        else f"http://192.168.0.100:8080/bestFromMap/{args.best_sample_size}"
    )
    score_queue_process_p = Process(target=score_queue_process, args=(score_queue,))
    score_queue_process_p.start()

    for i in range(round(num_instances* 4 if args.mode == "train" else num_instances)):

        queueProcess = Process(
            target=fetch_network_genome,
            args=(
                api_url,
                queue,
                substrate,
            ),
            daemon=True,
        )
        queueProcess.start()

    # while True:

    processes = []

    for i in range(num_instances):

        p = Process(
            target=simulation,
            args=(queue, score_queue, args, use_action_coords, i),
            daemon=True,
        )
        p.start()

        processes.append(p)

    for p in processes:
        p.join()
        p.close()


if __name__ == "__main__":
    main()
