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
from stageLengthMap import stageLengthMap

# from Network import CPPNConnectionQuery, NetworkProcessor, Substrate, TaskNetwork


# def fetch_network_genome(api_url):
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return json_to_network_genome(response.text)
#     else:
#         raise Exception(f"Failed to fetch network genome from {api_url}. Status code: {response.status_code}")
def simulate_environment(
    network_processor: NetworkProcessor,
    env: gym_super_mario_bros.SuperMarioBrosEnv,
    render: bool,
):
    scale = 1/16
    width = round(256 * scale)
    height = round(240 * scale)
    bias_coords = [(0, 0, -1.5)]
    input_coords = [
        (x, y, -1)
        for x in np.linspace(-1, 1, width)
        for y in np.linspace(-1, 1, height)  # for z in np.linspace(-.1, .1, 3)
    ]
    # previous_outputs = [(x, 0, -.5) for x in np.linspace(-1, 1, 12)]
    hidden_coords = [
        [
            (x, y, z)
            for x in np.linspace(-1, 1, round(width / 2))
            for y in np.linspace(-1, 1, round(height / 2))
        ]
        for z in np.linspace(-0.9, 0.9, round(30))
    ]
    # for z in np.linspace(-.5, .5, round(3))
    # data_dim = 3
    # top_k = 10
    # patch_size =7
    # patch_stride = 4
    # output_dim =3
    # query_dim =4
    # in_dim = data_dim * patch_size ** 2
    # out_dim = query_dim
    # [
    #     (x, y, z)
    #     for x in np.linspace(-1, 1, round(12))
    #     for y in np.linspace(-1, 1, round(12))
    #     for z in np.linspace(-.9, .9, round(2))
    # ]
    status :str = "small"	#Mario's status, i.e., {'small', 'tall', 'fireball'}
    output_width = 12
    output_height = 36
    output_coords = [(x, y, 1) for x in np.linspace(-1, 1, output_width) for y in np.linspace(-1, 1, output_height)]
    substrate = Substrate(input_coords, hidden_coords, output_coords, bias_coords)
    cppn_query_instance = CPPNConnectionQuery(network_processor, 3.0, 0.2)
    network = TaskNetwork2(substrate, cppn_query_instance)
    state: np.ndarray = env.reset()
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
    for step in range(20 * 200 * 8):
        image = (rescale(rgb2gray(state), scale) / 127.5) - 1
        # print(image.shape)
        torch_input = torch.from_numpy(image.flatten()).float()
        action_values : np.ndarray = network.forward(torch_input).reshape(output_width, output_height)
        
        # K = 3
        # top_values, top_indices = torch.topk(action_values, K, dim=1)
        # result = torch.zeros_like(action_values)
        # result.scatter_(1, top_indices, top_values)
        # action_probabilities = result.sum(axis=1).softmax(dim=-1)  # softmax(action_values)
        # * action_values.softmax(dim=0)
        # print(action_values)
        # print(action_values.softmax(dim=1))

        action_probabilities = (action_values.softmax(dim=0)).sum(dim=1) #.softmax(dim=-1)
        action = torch.argmax(action_probabilities)
        # print(action_probabilities)
        # if (action_probabilities[action.item()] < 0.1):
        # if render:
        #     print("action < 0.5")
        # action = torch.tensor(0)

        state, reward, done, info = env.step(action.item())
        cum_reward += reward
        x_pos = info["x_pos"]
        y_pos = info["y_pos"]
        movement_threshold = 16  # Define a threshold for movement reset
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
            speed_sum += abs(x_pos - x_pos_prev_movement)
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
    if info["flag_get"]:
        info["x_pos"] = stageLengthMap[(int(info["world"]), int(info["stage"]))]
    return info, cum_reward, average_speed, average_jump_count, average_fall_count, average_small_status_count, average_tall_status_count, average_fireball_status_count


def fetch_network_genome(api_url, queue: Queue):
    while True:
        # print("fetching")
        response = requests.get(api_url)
        if response.status_code == 200:
            data = json.loads(response.text)
            # print(data)
            if data["exists"]:
                network_genome = json_to_network_genome(data)
                # print("Network genome found " + str(data["id"]))
                queue.put([data["id"], network_genome])
            else:
                print("No network genome found - sleeping for 1 second")
                time.sleep(1)
        else:
            print(
                f"Failed to fetch network genome from {api_url}. Status code: {response.status_code}"
            )
            time.sleep(1)


def simulation(queue: Queue, render: bool):
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    while not queue.empty():
        data = queue.get()
        network_genome = data[1]
        # print("Building network " + str(data[0]))
        network_builder = NetworkBuilder(DefaultActivationFunctionMapper())
        network_processor_factory = NetworkProcessorFactory(
            network_builder, False, 1, 0.01
        )
        network_processor = network_processor_factory.createProcessor(network_genome)
        # print("starting simulation " + str(data[0]))
        info, cum_reward, average_speed, average_jump_count, average_fall_count, average_small_status_count, average_tall_status_count, average_fireball_status_count = (
            simulate_environment(network_processor, env, render)
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
            },
        )
        env.reset()

    print("finish?")
    env.close()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_instances', type=int, default=8,
                        help='number of instances to run')
    parser.add_argument('--should_render', type=bool, default=False,
                        help='render the environment')

    args = parser.parse_args()

    num_instances = args.num_instances
    should_render = args.should_render
    print(should_render)
    print(num_instances)
    manager = Manager()
    queue = manager.Queue(10)
    api_url = "http://192.168.0.100:8080/networkGenome"
    queueProcess = Process(
        target=fetch_network_genome,
        args=(
            api_url,
            queue,
        ),
    )
    queueProcess.start()

    while True:

        processes = []

        for i in range(num_instances):

            p = Process(target=simulation, args=(queue, should_render))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
