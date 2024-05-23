import json
import time
from urllib import request
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from multiprocessing import Manager, Process, Queue
import numpy as np
import requests
from skimage.transform import rescale
import torch
from Network import (
    CPPNConnectionQuery,
    DefaultActivationFunctionMapper,
    NetworkBuilder,
    NetworkProcessor,
    NetworkProcessorFactory,
    Substrate,
    TaskNetwork,
    TaskNetwork2,
    json_to_network_genome,
)
from SelfAttention import SelfAttention, calculate_patches
from stageLengthMap import stageLengthMap


def simulate_environment(
    network: TaskNetwork2,
    env: gym_super_mario_bros.SuperMarioBrosEnv,
    render: bool,
    scale: float,
    output_width: int,
    output_height: int,
    input_width: int,
    input_height: int,
    input_depth: int,
    color_channels: int,
):

    status: str = "small"  # Mario's status, i.e., {'small', 'tall', 'fireball'}

    active_state: np.ndarray = env.reset()
    done = False
    x_pos_prev = 40
    y_pos_prev = 0
    no_movement_count = 0
    cum_reward = 0
    action_values = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    speed_sum = 0
    tick_count = 0
    average_speed = 0
    average_jump_count = 0
    average_fall_count = 0
    average_small_status_count = 0
    average_tall_status_count = 0
    average_fireball_status_count = 0
    jump_count = 0
    fall_count = 0
    small_status_count = 0
    tall_status_count = 0
    fireball_status_count = 0
    x_pos_prev_movement = 40
    action_change_count = 0
    prev_action = 0
    image_input_history = [
        torch.zeros((input_height, input_width, color_channels)).flatten() for _ in range(input_depth)
    ]

    def update_image_input_history(new_image, history, max_size):
        """
        Adds a new image to the history and removes the oldest image to maintain the max size.

        :param new_image: The new image to be added.
        :param history: The list of historical images.
        :param max_size: The maximum size of the history list.
        :return: Updated history list.
        """
        if len(history) >= max_size:
            history.pop(0)  # Remove the oldest element
        history.append(new_image)  # Add the new element
        return history
    if color_channels == 1:
        image = (rescale(rgb2gray(active_state), scale) / 127.5) - 1
        torch_input = torch.from_numpy(image).permute(2,0, 1).flatten().float()
    elif color_channels == 3:
        image = (rescale(active_state, scale, channel_axis=2) / 127.5) - 1
        torch_input = torch.from_numpy(image).flatten().float()
    # print(image.shape)
    # torch_input = torch.from_numpy(image).permute(2,0, 1).flatten().float()
    image_input_history = update_image_input_history(
        torch_input, image_input_history, input_depth
    )
    large_flatten_tensor = torch.cat([tensor for tensor in image_input_history])
    # Example usage:
    # new_image = torch.zeros((height, width)).flatten()  # Replace with actual new image tensor
    # image_input_history = update_image_input_history(new_image, image_input_history, input_depth)
    while True:

        # Join each tensor in image_input_history into one large flatten tensor

        action_values: np.ndarray = network.forward(large_flatten_tensor).reshape(
            output_height, output_width
        )

        # K = 3
        # top_values, top_indices = torch.topk(action_values, K, dim=1)
        # result = torch.zeros_like(action_values)
        # result.scatter_(1, top_indices, top_values)
        # action_probabilities = result.sum(axis=1).softmax(dim=-1)  # softmax(action_values)
        # * action_values.softmax(dim=0)
        # print(action_values)
        # print(action_values.softmax(dim=1))
        # action_values.softmax(dim=0) *
        action_probabilities = (action_values.softmax(dim=1)).sum(
            dim=0
        )  # .softmax(dim=-1)
        action = torch.argmax(action_probabilities)
        if action != prev_action:
            action_change_count += 1
        prev_action = action
        # print(action_probabilities)
        # if (action_probabilities[action.item()] < 0.1):
        # if render:
        #     print("action < 0.5")
        # action = torch.tensor(0)

        state, reward, done, info = env.step(action.item())
        
        if tick_count % 20 * 2 == 0:
            # image = (rescale(rgb2gray(active_state), scale) / 127.5) - 1
            if color_channels == 1:
                image = (rescale(rgb2gray(active_state), scale) / 127.5) - 1
                torch_input = torch.from_numpy(image).permute(2,0, 1).flatten().float()
            elif color_channels == 3:
                image = (rescale(active_state, scale, channel_axis=2) / 127.5) - 1
                torch_input = torch.from_numpy(image).flatten().float()
            # print(image.shape)
            
            active_state = state
            image_input_history = update_image_input_history(
                torch_input, image_input_history, input_depth
            )
            large_flatten_tensor = torch.cat([tensor for tensor in image_input_history])
        cum_reward += reward
        x_pos = info["x_pos"]
        y_pos = info["y_pos"]
        movement_threshold = 32  # Define a threshold for movement reset
        if abs(x_pos - x_pos_prev) < movement_threshold:
            no_movement_count += 1
        else:
            x_pos_prev = x_pos

            no_movement_count = 0
        if no_movement_count >= 20 * 40:
            break
        if y_pos > y_pos_prev:
            jump_count += 1
        if y_pos < y_pos_prev:
            fall_count += 1
        # Measure the average speed of Mario
        if info["status"] == "small":
            small_status_count += 1
        if info["status"] == "tall":
            tall_status_count += 1
        if info["status"] == "fireball":
            fireball_status_count += 1
        if tick_count > 0:
            speed_sum += min(4, abs(x_pos - x_pos_prev_movement))
        tick_count += 1
        if tick_count > 0:
            average_speed = speed_sum / tick_count
            average_jump_count = (jump_count + fall_count) / tick_count
            average_fall_count = fall_count / tick_count
            average_small_status_count = small_status_count / tick_count
            average_tall_status_count = tall_status_count / tick_count
            average_fireball_status_count = fireball_status_count / tick_count
        else:
            average_speed = 0
            average_jump_count = 0
            average_fall_count = 0
        # print("Average speed: ", average_speed)
        # print("Jump count: ", jump_count)
        x_pos_prev_movement = x_pos
        y_pos_prev = y_pos
        if done or reward < -10:
            break
        # print(render)
        if render:
            # print(network.input_hidden_weights)
            # print(network.hidden_output_weights)
            # print(action)
            # print(action_probabilities[action.item()])
            # print(action_probabilities)
            env.render()
    # if info["flag_get"]:
    #     info["x_pos"] = stageLengthMap[(int(info["world"]), int(info["stage"]))]
    if info["stage"] == 4 and info["x_pos"] < cum_reward:
        cum_reward = info["x_pos"] - (400 - info["time"])
    return (
        info,
        cum_reward,
        average_speed,
        average_jump_count,
        average_fall_count,
        average_small_status_count,
        average_tall_status_count,
        average_fireball_status_count,
        action_change_count,
    )


def fetch_network_genome(api_url, queue: Queue, substrate: Substrate):

    while True:
        # print("fetching")
        response = requests.get(api_url)

        if response.status_code == 200:
            data = json.loads(response.text)
            # print(data)
            if data["exists"]:
                network_genome = json_to_network_genome(data)
                network_builder = NetworkBuilder(DefaultActivationFunctionMapper())
                network_processor_factory = NetworkProcessorFactory(
                    network_builder, False, 1, 0.01
                )
                network_processor = network_processor_factory.createProcessor(
                    network_genome
                )
                cppn_query_instance = CPPNConnectionQuery(network_processor, 3.0, 0.3)
                network = TaskNetwork2(substrate, cppn_query_instance)
                # print("Network genome found " + str(data["id"]))
                queue.put([data["id"], network])
            else:
                print("No network genome found - sleeping for 1 second")
                time.sleep(0.1)
        else:
            print(
                f"Failed to fetch network genome from {api_url}. Status code: {response.status_code}"
            )
            time.sleep(1)


def simulation(
    queue: Queue,
    render: bool,
    scale: float,
    output_width: int,
    output_height: int,
    input_width: int,
    input_height: int,
    input_depth: int,
    color_channels: int,
):
    env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0")
    # env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    while True:
        data = queue.get()
        network: TaskNetwork2 = data[1]

        # print("starting simulation " + str(data[0]))
        (
            info,
            cum_reward,
            average_speed,
            average_jump_count,
            average_fall_count,
            average_small_status_count,
            average_tall_status_count,
            average_fireball_status_count,
            action_change_count,
        ) = simulate_environment(
            network,
            env,
            render,
            scale,
            output_width,
            output_height,
            input_width,
            input_height,
            input_depth,
            color_channels,
        )
        # print(info)

        requests.post(
            "http://192.168.0.100:8080/score",
            json={
                "id": data[0],
                "x": int(info["x_pos"]),
                "y": int(info["y_pos"]),
                "score": int(info["score"]),
                "time": int(info["time"]),
                "reward": float(cum_reward),
                "stage": int(info["stage"]),
                "world": int(info["world"]),
                "coins": int(info["coins"]),
                "life": int(info["life"]),
                "flagGet": bool(info["flag_get"]),
                "status": str(info["status"]),
                "averageSpeed": float(average_speed),
                "averageJumpCount": float(average_jump_count),
                "averageFallCount": float(average_fall_count),
                "averageSmallStatusCount": float(average_small_status_count),
                "averageTallStatusCount": float(average_tall_status_count),
                "averageFireballStatusCount": float(average_fireball_status_count),
                "actionChangeCount": float(action_change_count),
            },
        )
        env.reset()

    print("finish?")
    env.close()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--num_instances", type=int, default=8, help="number of instances to run"
    )
    parser.add_argument(
        "--should_render", type=bool, default=False, help="render the environment"
    )

    args = parser.parse_args()
    color_channels = 1
    scale = 1 / 16
    width = round(256 * scale)
    height = round(240 * scale)
    input_depth = 10
    bias_coords = [(0, 0, -2)]
    
    input_coords = [
        (y, x, z)
        for y in np.linspace(-1, 1, height)
        for x in np.linspace(-1, 1, width)
        for z in np.linspace(-1, -0.9, input_depth * color_channels)
    ]
    # previous_outputs = [(x, 0, -.5) for x in np.linspace(-1, 1, 12)]
    attention_coords = [
        (y, x, -0.5)
        for y in np.linspace(-1, 1, height)
        for x in np.linspace(-1, 1, width)
    ]
    # for y in np.linspace(-1, 1, 1)
    hidden_coords = [
        [(y, x, z) for y in np.linspace(-1, 1, 6) for x in np.linspace(-1, 1, 8)]
        for z in np.linspace(-0.9, 0.9, round(30))
    ]
    output_width = 12
    output_height = 48
    output_coords = [
        (y, x, 1)
        for y in np.linspace(-1, 1, output_height)
        for x in np.linspace(-1, 1, output_width)
    ]
    substrate = Substrate(input_coords, hidden_coords, output_coords, bias_coords)
    num_instances = args.num_instances
    should_render = args.should_render
    print(should_render)
    print(num_instances)
    manager = Manager()
    queue = manager.Queue(num_instances * 4)
    api_url = "http://192.168.0.100:8080/networkGenome"
    for i in range(round(num_instances * 2)):
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

    while True:

        processes = []

        for i in range(num_instances):

            p = Process(
                target=simulation,
                args=(
                    queue,
                    should_render,
                    scale,
                    output_width,
                    output_height,
                    width,
                    height,
                    input_depth,
                    color_channels,
                ),
                daemon=True,
            )
            p.start()

            processes.append(p)
            if should_render:
                time.sleep(0.5)

        for p in processes:
            p.join()
            p.close()
