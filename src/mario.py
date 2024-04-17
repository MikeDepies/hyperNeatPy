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
    width = 16
    height = 15

    input_coords = [
        (x, y, -1) for x in np.linspace(-1, 1, width) for y in np.linspace(-1, 1, height)
    ]
    hidden_coords = [
        (x, y, 0.0)
        for x in np.linspace(-1, 1, round(width/2))
        for y in np.linspace(-1, 1, round(height/2))
    ]
    output_coords = [(x, y, 1) for x in np.linspace(-1, 1, 4) for y in np.linspace(-1, 1, 3)]
    substrate = Substrate(input_coords, hidden_coords, output_coords)
    cppn_query_instance = CPPNConnectionQuery(network_processor, 3.0, 0.6)
    network = TaskNetwork(substrate, cppn_query_instance)
    state: np.ndarray = env.reset()
    done = False
    x_pos_prev = 0
    y_pos_prev = 0
    no_movement_count = 0
    cum_reward = 0
    for step in range(20 * 200):
        image = rescale(rgb2gray(state), 1 / 16)
        action_values = network.forward(torch.from_numpy(image.flatten()).float()).flatten()
        softmax = torch.nn.Softmax(dim=0)
        action_probabilities = softmax(action_values)
        action = torch.argmax(
            action_probabilities
        )
        state, reward, done, info = env.step(action.item())
        cum_reward += reward
        x_pos = info["x_pos"]
        y_pos = info["y_pos"]
        if x_pos == x_pos_prev and y_pos == y_pos_prev:
            no_movement_count += 1
        else:
            no_movement_count = 0
        if no_movement_count >= 20*20:
            break
        x_pos_prev = x_pos
        y_pos_prev = y_pos

        if done or reward < -10:
            break
        # print(render)
        if render:
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
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
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
    queue = manager.Queue(20)
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

        for i in range(10):

            p = Process(target=simulation, args=(queue, i < 0))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
