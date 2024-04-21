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
    json_to_network_genome,
)

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
    scale = 8
    width = 256 / scale
    height = 240 / scale
    bias_coords = [(0,0,-.5)]
    input_coords = [
        # (x, y, -1) for x in np.linspace(-1, 1, width) for y in np.linspace(-1, 1, height)
        (x, y, -1 + z) for x in np.linspace(-1, 1, width) for y in np.linspace(-1, 1, height) for z in np.linspace(-.1, .1, 3)
    ]
    # previous_outputs = [(x, 0, -.5) for x in np.linspace(-1, 1, 12)]
    hidden_coords = [
        (x, y, 0.0)
        for x in np.linspace(-1, 1, round(10))
        for y in np.linspace(-1, 1, round(10))
    ]
    output_coords = [(x, 0, 1) for x in np.linspace(-1, 1, 12) ]
    substrate = Substrate(input_coords, hidden_coords, output_coords, bias_coords)
    cppn_query_instance = CPPNConnectionQuery(network_processor, 3.0, 0.0)
    network = TaskNetwork(substrate, cppn_query_instance)
    state: np.ndarray = env.reset()
    done = False
    x_pos_prev = 40
    y_pos_prev = 0
    no_movement_count = 20*10
    cum_reward = 0
    action_values = torch.tensor([0,0,0,0,0,0,0,0,0, 0, 0, 0])
    for step in range(20 * 200):
        # rgb2gray(state)
        image = (rescale(state, 1 / scale, channel_axis=2) / 255)
        # print(image)
        torch_input = torch.from_numpy(image.flatten()).float()
        action_values = network.forward(torch_input).flatten()
        # softmax = torch.nn.Softmax(dim=0)
        action_probabilities = action_values#softmax(action_values)
        action = torch.argmax(
            action_probabilities
        )
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
            y_pos_prev = y_pos
            no_movement_count = 0
        if no_movement_count >= 20*20:
            break
       

        if done or reward < -10:
            break
        # print(render)
        if render:
            # print(network.input_hidden_weights)
            # print(network.hidden_output_weights)
            print(action)
            print(action_probabilities[action.item()])
            print(action_probabilities)
            env.render()

    return info, cum_reward


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


def base():
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    state: np.ndarray = env.reset()
    done = False
    for step in range(5000):
        if done:
            state = env.reset()
        image = state
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        env.render()

    env.close()


def simulation(queue: Queue, render: bool):
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
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
        info, cum_reward = simulate_environment(network_processor, env, render)
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
            },
        )
        env.reset()

    print("finish?")
    env.close()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    manager = Manager()
    queue = manager.Queue(30)
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

        for i in range(20):

            p = Process(target=simulation, args=(queue, i < 1))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
