import argparse
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


def parseArgs():
    parser = argparse.ArgumentParser(description="Melee")

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
        
    )


def createMelee(args: MeleeArgs, instance_num: int):
    if args.mode == "stream":
        console = Console(
            path=args.dolphin_executable_path,
            slippi_port=args.dolphin_port + instance_num,
            blocking_input=False,
            polling_mode=False,
        )
    else:
        console = Console(
            path=args.dolphin_executable_path,
            slippi_port=args.dolphin_port + instance_num,
            blocking_input=True,
            polling_mode=False,
            setup_gecko_codes=True,
            gfx_backend="Null",
            use_exi_inputs=True,
            enable_ffw=True,
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
        self.prev_input = (.5, .5, .5, .5, 0.0, False, False, False, False)

    def player(self, game_state: GameState) -> PlayerState:
        return game_state.players[self.agent_configuration.port]

    def create_action(self, input: Tensor):
        if self.task_network is None:
            return
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
        center_advantage: float
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
        center_advantage_delta: float
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
            agent1_score_delta.action = player1.action
            
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
            self.agent_scores[
                delta_score.agent.agent_configuration.port
            ].center_advantage += delta_score.center_advantage_delta
            self.agent_scores[
                delta_score.agent.agent_configuration.port
            ].unique_actions.add(delta_score.action)
        return self.agent_scores


class MeleeConfiguration:
    def __init__(self, agents: List[AgentConfiguration], stage: melee.Stage):
        self.agents = agents
        self.stage = stage


from enum import Enum

def stageToString(stage : melee.Stage):
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

def characterToString(character : melee.Character):
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

class SimulationState(Enum):
    RUNNING = 0
    GAME_OVER = 1
    MENU = 2


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
        

    def set_config(self, melee_config: MeleeConfiguration):
        self.melee_config = melee_config

    def simulation_step(
        self,
        game_state: GameState,
        game_state_evaluator: GameStateEvaluator,
        menu_helper: melee.menuhelper.MenuHelper,
    ):
        if game_state.menu_state in [
            melee.enums.Menu.IN_GAME,
            melee.enums.Menu.SUDDEN_DEATH,
        ]:
            agent_scores = game_state_evaluator.score_agents(game_state)
            if self.isGameOver(game_state, agent_scores)[0]:
                return (game_state_evaluator.agent_scores, SimulationState.GAME_OVER)
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

    def isGameOver(self, game_state: GameState, agent_scores: List[AgentScore]):
        players: List[Tuple[PlayerState, int]] = list(
            map(lambda x: (agent_scores[x].agent.player(game_state), x), agent_scores)
        )
        for player, index in players:
            if player.stock <= 0:
                return (True, index)
        return (False, None)

    def handle_game_step(self, game_state: GameState, agents: List[Agent]):
        task_input: Tensor
        if self.use_action_coords:
            input_tensor, input_action_tensor = game_state_to_tensor(
                game_state, agents[0].player(game_state), agents[1].player(game_state)
            )
            task_input = torch.cat(
                (input_tensor.flatten(), input_action_tensor.flatten())
            )
        else:
            input_tensor = game_state_to_tensor_action_normalized(
                game_state, agents[0].player(game_state), agents[1].player(game_state)
            )
            task_input = input_tensor.flatten()
        output = agents[0].create_action(task_input)
        if output is not None:
            buttons, analog, c_analog = output_tensor_to_controller_tensors(
                torch.sigmoid(output)
            )
            
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
            main_stick_x, main_stick_y = self.controller_helper.processAnalog(analog)
            c_analog_x, c_analog_y = self.controller_helper.processAnalog(c_analog)
            # print(f"main stick x: {main_stick_x}")
            # print(f"main stick y: {main_stick_y}")
            # print(f"c stick x: {c_analog_x}")
            # print(f"c stick y: {c_analog_y}")
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
                agents[0].controller,
            )
            new_input = (main_stick_x, main_stick_y, c_analog_x, c_analog_y, left_shoulder, press_a, press_b, press_y, press_z)
            input_delta = 0
            input_delta = sum(1 for new, prev in zip(new_input, agents[0].prev_input) if new != prev)
            agents[0].prev_input = new_input
            agents[0].input_count += input_delta


def simulation(
    queue: Queue, score_queue: Queue, args: MeleeArgs, use_action_coords: bool, instance_num: int
):
    meleeCore = createMelee(args, instance_num)
    meleeCore.run()
    signal_handler = MeleeSignalHandler(meleeCore)
    signal.signal(signal.SIGINT, signal_handler.signal_handler)
    print("Connecting...")
    time.sleep(5)
    print("Connecting!")
    meleeCore.connect()

    menu_helper = melee.menuhelper.MenuHelper
    print("simulation starting...")
    id: str
    network: TaskNetwork2
    agent_config: AgentConfiguration
    cpu_config: AgentConfiguration
    stage : melee.Stage
    # fetch from queue
    while True:
        # print("get next network")
        (id, network, agent_config, cpu_config, stage) = queue.get()
        agent_configuration_list = [
            agent_config,
            cpu_config,
        ]
        melee_config = MeleeConfiguration(
            agent_configuration_list, stage
        )
        meleeSimulation = MeleeSimulation(meleeCore, melee_config, use_action_coords)
        agents = [
                Agent(meleeCore, agent_config, meleeCore.controller, network),
                Agent(meleeCore, cpu_config, meleeCore.controller_opponent, None),
            ]
        game_state_evaluator = GameStateEvaluator(
            meleeCore,
            agents,
            GameStateDeltaProcessor(),
        )
        agent_score: AgentScore
        first_step = True
        print(f"start loop {instance_num}")
        if agent_config.character == melee.Character.CPTFALCON and cpu_config.character == melee.Character.CPTFALCON:
            continue
        while True:
            game_state = meleeCore.next_step()
            if first_step:
                print(f"({instance_num}) agent {agent_config.character} vs cpu {cpu_config.character}")
                first_step = False
            if game_state.frame > 60*60*4: #agent_config.character == melee.Character.CPTFALCON and cpu_config.character == melee.Character.CPTFALCON:
                if game_state.frame % (60 * 10) == 0 and game_state.menu_state == melee.Menu.IN_GAME:
                    print(f"({instance_num})  agent {agent_config.character} vs cpu {cpu_config.character} frame: {game_state.frame} agent x: {agents[0].player(game_state).x} cpu x: {agents[1].player(game_state).x} agent percent: {agents[0].player(game_state).percent}")
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
                game_state, game_state_evaluator, menu_helper
            )
            agent_score = score[1]
            if state == SimulationState.GAME_OVER:
                # print("Game Over")
                meleeCore.controller.release_all()
                break
        # print((id, agent_score.kill_count, agent_score.death_count, agent_score.damage_dealt, agent_score.damage_received))
        score_dict = {
            "id": id,
            "kill_count": agent_score.kill_count,
            "death_count": agent_score.death_count,
            "damage_dealt": agent_score.damage_dealt,
            "damage_received": agent_score.damage_received,
            "center_advantage": agent_score.center_advantage,
            "unique_action_count": len(agent_score.unique_actions),
            "total_frames" : int(game_state.frame),
            "input_count": agents[0].input_count,
            "cpu_level": cpu_config.cpu_level,
            "stage": stageToString(melee_config.stage),
            "character" : characterToString(agent_config.character),
            "opponent_character" : characterToString(cpu_config.character),

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
            cppn_query_instance = CPPNConnectionQuery(network_processor, 3.0, 0.6)
            network = TaskNetwork2(substrate, cppn_query_instance)
            queue.put(
                [
                    data["id"],
                    network,
                    task.agent_task.agent_config,
                    task.cpu_task.agent_config,
                    task.stage
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
                "totalFrames": score["total_frames"],
                "cpuLevel": score["cpu_level"],
                "stage": score["stage"],
                "character": score["character"],
                "opponentCharacter": score["opponent_character"],
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

def game_state_to_tensor(
    game_state: GameState, agent_player: PlayerState, opponent_player: PlayerState
):
    input_tensor = torch.zeros((2, 10))
    input_action_tensor = torch.zeros((2, 386))
    for i, player in enumerate([agent_player, opponent_player]):
        input_tensor[i, 0] = player.percent / 100
        input_tensor[i, 1] = player.stock / 4
        input_tensor[i, 2] = player.x / 100
        input_tensor[i, 3] = player.y / 100
        input_tensor[i, 4] = 1 if player.facing == True else -1
        input_tensor[i, 5] = player.jumps_left / 2
        input_tensor[i, 6] = player.shield_strength / 60
        input_tensor[i, 7] = stageToInt(game_state.stage) / 6
        input_tensor[i, 8] = characterToInt(player.character) / 26
        input_tensor[i, 9] = player.action_frame / 60
        input_action_tensor[i, min(385, player.action.value)] = 1

    return input_tensor, input_action_tensor


def game_state_to_tensor_action_normalized(
    game_state: GameState, agent_player: PlayerState, opponent_player: PlayerState
):
    input_tensor = torch.zeros((2, 11))
    # input_action_tensor = torch.zeros((2, 386))
    for i, player in enumerate([agent_player, opponent_player]):
        input_tensor[i, 0] = player.percent / 100
        input_tensor[i, 1] = player.stock / 4
        input_tensor[i, 2] = player.x / 100
        input_tensor[i, 3] = player.y / 100
        input_tensor[i, 4] = 1 if player.facing == True else -1
        input_tensor[i, 5] = player.jumps_left / 2
        input_tensor[i, 6] = player.shield_strength / 60
        input_tensor[i, 7] = min(385, player.action.value) / 385
        input_tensor[i, 8] = stageToInt(game_state.stage) / 6
        input_tensor[i, 9] = characterToInt(player.character) / 26
        input_tensor[i, 10] = player.action_frame / 60

    return input_tensor


def output_tensor_to_controller_tensors(output_tensor: Tensor):
    buttons = output_tensor[:, :5].reshape(1, 5)
    analog = output_tensor[:, 5:105].reshape(10, 10)
    c_analog = output_tensor[:, 105:205].reshape(10, 10)
    return buttons, analog, c_analog


def main():
    args = parseArgs()
    use_action_coords = True
    width = 10 if use_action_coords else 11
    height = 2
    action_width = 386
    action_height = 2
    bias_coords = [(0, 0, .3)]
    input_coords = [
        (y, x, -1)
        for y in np.linspace(-1, 1, height)  # for z in np.linspace(-.1, .1, 3)
        for x in np.linspace(-1, 1, width)
    ]
    action_coords = [
        (y, x, -0.9)
        for y in np.linspace(-1, 1, action_height)
        for x in np.linspace(-1, 1, action_width)
    ]
    # for y in np.linspace(-1, 1, 1)
    hidden_coords = [
        [(y, x, z) for y in np.linspace(-1, 1, 12) for x in np.linspace(-1, 1, 16)]
        for z in np.linspace(-0.9, 0.0, round(8))
    ]
    output_width = 5
    output_height = 1
    analog_width = 10
    analog_height = 10
    analog_coords = [
        (y, x, 0.6)
        for y in np.linspace(-1, 0, analog_height)
        for x in np.linspace(-1, 0, analog_width)
    ]
    c_analog_coords = [
        (y, x, 0.6)
        for y in np.linspace(0, 1, analog_height)
        for x in np.linspace(0, 1, analog_width)
    ]
    output_coords = [
        (y, x, 1)
        for y in np.linspace(-1, 1, output_height)
        for x in np.linspace(-1, 1, output_width)
    ]

    all_input_coords = (
        input_coords + action_coords if use_action_coords else input_coords
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
    queue = manager.Queue(num_instances * 2)
    score_queue = manager.Queue(num_instances * 5)
    api_url = "http://192.168.0.100:8080/networkGenome" if mode == "train" else "http://192.168.0.100:8080/bestFromMap/300"
    score_queue_process_p = Process(target=score_queue_process, args=(score_queue,))
    score_queue_process_p.start()

    for i in range(round(num_instances)):
        
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
