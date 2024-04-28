import json
import torch
from NetworkGenome import ConnectionGenome, NodeGenome, NodeType, ActivationFunction, NetworkGenome
from typing import Callable, List, Dict
from math import exp, tanh, sin, cos, sqrt

from SelfAttention import SelfAttention
ActivationFunctionType = Callable[[float], float]


class Node:
    def __init__(self, id: int, type: NodeType, activation_function: ActivationFunctionType, bias: float):
        self.id = id
        self.type = type
        self.activation_function = activation_function
        self.bias = bias
        self.input_value = 0.0
        self.output_value = 0.0

    def activate(self):
        if self.type == NodeType.INPUT:
            self.output_value = self.input_value
        else:
            self.output_value = self.activation_function(self.input_value + self.bias)

class Connection:
    def __init__(self, input_node_id: int, output_node_id: int, weight: float):
        self.input_node_id = input_node_id
        self.output_node_id = output_node_id
        self.weight = weight

class Network:
    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.nodes = nodes
        self.connections = connections

class DefaultActivationFunctionMapper:
    def map(self, activationFunction: ActivationFunction) -> ActivationFunctionType:
        return {
            ActivationFunction.IDENTITY: lambda x: max(-1.0, min(1.0, x)),
            ActivationFunction.SIGMOID: lambda x: max(-1.0, min(1.0, 1 / (1 + exp(-x)))),
            ActivationFunction.TANH: lambda x: max(-1.0, min(1.0, tanh(x))),
            ActivationFunction.RELU: lambda x: max(-1.0, min(1.0, max(0.0, x))),
            ActivationFunction.GAUSSIAN: lambda x: max(-1.0, min(1.0, exp(-x ** 2.0))),
            ActivationFunction.SINE: lambda x: max(-1.0, min(1.0, sin(x))),
            ActivationFunction.COS: lambda x: max(-1.0, min(1.0, cos(x))),
            ActivationFunction.ABS: lambda x: max(-1.0, min(1.0, abs(x))),
            ActivationFunction.STEP: lambda x: max(-1.0, min(1.0, 0.0 if x < 0 else 1.0))
        }[activationFunction]

class NetworkBuilder:
    def __init__(self, activation_function_mapper: DefaultActivationFunctionMapper):
        self.activation_function_mapper = activation_function_mapper

    def build_network_from_genome(self, genome: NetworkGenome) -> Network:
        nodes = [Node(node_genome.id, node_genome.type, 
                      self.activation_function_mapper.map(node_genome.activation_function), 
                      node_genome.bias) for node_genome in genome.node_genomes]

        connections = [Connection(connection_gene.input_node, connection_gene.output_node, 
                                   connection_gene.weight) for connection_gene in genome.connection_genes if connection_gene.enabled]

        return Network(nodes, connections)
class NetworkProcessor:
    def feedforward(self, input_values: List[float]) -> List[float]:
        raise NotImplementedError
class NetworkProcessorSimple(NetworkProcessor):
    def __init__(self, network: Network):
        self.network = network
        self.output_nodes = [node for node in network.nodes if node.type == NodeType.OUTPUT]
        self.input_nodes = [node for node in network.nodes if node.type == NodeType.INPUT]
        self.sorted_nodes = sorted(network.nodes, key=lambda node: node.type.value)
        self.input_connections_by_output_node_id = {node.id: [conn for conn in network.connections if conn.output_node_id == node.id] for node in network.nodes}
        self.node_map = {node.id: node for node in network.nodes}

    def feedforward(self, input_values: List[float]) -> List[float]:
        for node in self.network.nodes:
            node.input_value = 0.0

        for index, node in enumerate(self.input_nodes):
            node.input_value = input_values[index]

        for node in self.sorted_nodes:
            for connection in self.input_connections_by_output_node_id.get(node.id, []):
                input_node = self.node_map.get(connection.input_node_id)
                if input_node:
                    node.input_value += input_node.output_value * connection.weight
            node.activate()

        return [node.output_value for node in self.output_nodes]

class NetworkProcessorStateful(NetworkProcessor):
    def __init__(self, network: Network, max_iterations: int = 10, convergence_threshold: float = 0.01):
        self.network = network
        self.output_nodes = [node for node in network.nodes if node.type == NodeType.OUTPUT]
        self.input_nodes = [node for node in network.nodes if node.type == NodeType.INPUT]
        self.sorted_nodes = sorted(network.nodes, key=lambda node: node.type.value)
        self.node_map = {node.id: node for node in network.nodes}
        self.input_connections_by_output_node_id = {node.id: [conn for conn in network.connections if conn.output_node_id == node.id] for node in network.nodes}
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def feedforward(self, input_values: List[float]) -> List[float]:
        self.reset_input_values()
        self.assign_input_values(input_values)

        previous_output_values = [node.output_value for node in self.output_nodes]
        iteration = 0
        converged = False

        while iteration < self.max_iterations and not converged:
            self.process_nodes(self.sorted_nodes, self.node_map, self.input_connections_by_output_node_id)

            current_output_values = [node.output_value for node in self.output_nodes]
            converged = self.is_converged(previous_output_values, current_output_values, self.convergence_threshold)
            if not converged:
                previous_output_values = current_output_values
            iteration += 1

        return [node.output_value for node in self.output_nodes]

    def reset_input_values(self):
        for node in self.network.nodes:
            node.input_value = 0.0

    def assign_input_values(self, input_values: List[float]):
        for index, node in enumerate(self.input_nodes):
            node.input_value = input_values[index]

    def process_nodes(self, sorted_nodes: List[Node], node_map: Dict[int, Node], input_connections_by_output_node_id: Dict[int, List[Connection]]):
        for node in sorted_nodes:
            for connection in input_connections_by_output_node_id.get(node.id, []):
                input_node = node_map.get(connection.input_node_id)
                if input_node:
                    node.input_value += input_node.output_value * connection.weight
            node.activate()

    def is_converged(self, previous_output_values: List[float], current_output_values: List[float], threshold: float) -> bool:
        return all(abs(prev - curr) < threshold for prev, curr in zip(previous_output_values, current_output_values))


class NetworkProcessorFactory:
    def __init__(self, networkBuilder: NetworkBuilder, cyclic: bool, maxIterations: int = 10, convergenceThreshold: float = 0.01):
        self.networkBuilder = networkBuilder
        self.cyclic = cyclic
        self.maxIterations = maxIterations
        self.convergenceThreshold = convergenceThreshold

    def createProcessor(self, genome: NetworkGenome) -> NetworkProcessor:
        if self.cyclic and NetworkGenomeTester().has_cyclic_connections(genome):
            return NetworkProcessorStateful(self.networkBuilder.build_network_from_genome(genome), self.maxIterations, self.convergenceThreshold)
        else:
            return NetworkProcessorSimple(self.networkBuilder.build_network_from_genome(genome))

class NetworkCycleTester:
    def __init__(self, network):
        self.network = network

    def has_cyclic_connections(self):
        visited = set()
        rec_stack = set()

        for node in self.network.nodes:
            if self.dfs(node.id, visited, rec_stack):
                return True
        return False

    def dfs(self, current_node_id, visited, rec_stack):
        stack = [current_node_id]

        while stack:
            node_id = stack[-1]

            if node_id not in visited:
                if node_id in rec_stack:
                    return True
                visited.add(node_id)
                rec_stack.add(node_id)
                child_nodes = [connection.output_node_id for connection in self.network.connections if connection.input_node_id == node_id]
                stack.extend(child_nodes)
            else:
                rec_stack.remove(node_id)
                stack.pop()
        return False

class NetworkGenomeTester:
    def has_cyclic_connections(self, genome):
        visited = set()
        rec_stack = set()

        for current_node_id in [node.id for node in genome.node_genomes]:
            if current_node_id not in visited:
                if self.is_cyclic_util(current_node_id, genome, visited, rec_stack):
                    return True
        return False

    def is_cyclic_util(self, start_node_id, genome, visited, rec_stack):
        stack = [(start_node_id, iter(self.get_child_node_ids(start_node_id, genome)))]

        while stack:
            node_id, iterator = stack[-1]

            if node_id not in visited:
                visited.add(node_id)
                rec_stack.add(node_id)

            cycle_detected = False
            for child_node_id in iterator:
                if child_node_id in rec_stack:
                    cycle_detected = True
                    break
                elif child_node_id not in visited:
                    stack.append((child_node_id, iter(self.get_child_node_ids(child_node_id, genome))))
                    cycle_detected = False
                    break

            if cycle_detected:
                return True
            elif not iterator:
                rec_stack.remove(node_id)
                stack.pop()

        return False

    def get_child_node_ids(self, node_id, genome):
        return [connection.output_node for connection in genome.connection_genes if connection.input_node == node_id and connection.enabled]

class CPPNConnectionQuery:
    def __init__(self, networkProcessor : NetworkProcessor, connection_magnitude_multiplier : float, connection_threshold : float):
        self.networkProcessor = networkProcessor
        self.connection_magnitude_multiplier = connection_magnitude_multiplier
        self.connection_threshold = connection_threshold

    def query(self, x1, y1, z1, x2, y2, z2, d):
        
        input_values = [x1, y1, z1, x2, y2, z2, d]
        output_values = self.networkProcessor.feedforward(input_values)
        sign = output_values[0] / abs(output_values[0]) if output_values[0] != 0 else 0
        output_abs = abs(output_values[0])
        if output_abs > self.connection_threshold:
            normalized_output = (output_abs - self.connection_threshold) / (1 - self.connection_threshold)
            connection_magnitude = self.connection_magnitude_multiplier * normalized_output * sign
        else:
            connection_magnitude = 0
        return connection_magnitude

def distance(coord1, coord2):
    return sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
class Substrate:
    def __init__(self, input_coords, hidden_coords, output_coords, bias_coords):
        """
        Initialize the substrate with 3D coordinates for input, hidden, and output layers.
        """
        self.input_coords = input_coords
        self.hidden_coords = hidden_coords
        self.output_coords = output_coords
        self.bias_coords = bias_coords

    def get_recurrent_layer_connections(self, layer1_coords, layer2_coords, cppn_query):
        weights = torch.zeros((len(layer1_coords), len(layer2_coords)))
        for i, coord1 in enumerate(layer1_coords):
            # for j, coord2 in enumerate(layer2_coords):
            weight = cppn_query.query(*coord1, *coord1, 0)
            weights[i, i] = weight #if abs(weight) > 1e-5 else 0.0  # Apply thresholding
        return torch.nn.Parameter(weights, requires_grad=False)

    def get_layer_connections(self, layer1_coords, layer2_coords, cppn_query):
        """
        Generate a matrix of connections (weights) between two layers using 3D coordinates.
        """
        weights = torch.zeros((len(layer1_coords), len(layer2_coords)))
        for i, coord1 in enumerate(layer1_coords):
            for j, coord2 in enumerate(layer2_coords):
                weight = cppn_query.query(*coord1, *coord2, distance(coord1, coord2))
                weights[i, j] = weight #if abs(weight) > 1e-5 else 0.0  # Apply thresholding
        return torch.nn.Parameter(weights, requires_grad=False)
class TaskNetwork(torch.nn.Module):
    def __init__(self, substrate : Substrate, cppn_query : CPPNConnectionQuery):
        super(TaskNetwork, self).__init__()
        self.substrate = substrate

        # Get connection matrices with weights potentially being zero
        self.input_hidden_weights = substrate.get_layer_connections(substrate.input_coords, substrate.hidden_coords, cppn_query)
        self.hidden_output_weights = substrate.get_layer_connections(substrate.hidden_coords, substrate.output_coords, cppn_query)
        self.output_hidden_weights = substrate.get_layer_connections(substrate.output_coords, substrate.hidden_coords, cppn_query)
        self.hidden_bias_weights = substrate.get_layer_connections(substrate.bias_coords, substrate.hidden_coords, cppn_query)
        self.output_bias_weights = substrate.get_layer_connections(substrate.bias_coords, substrate.output_coords, cppn_query)
        self.hidden_recurrent_weights = substrate.get_recurrent_layer_connections(substrate.hidden_coords, substrate.hidden_coords, cppn_query)
        self.output_recurrent_weights = substrate.get_recurrent_layer_connections(substrate.output_coords, substrate.output_coords, cppn_query)
        self.outputs = torch.zeros(self.output_bias_weights.shape[0], self.output_bias_weights.shape[1])
        self.hidden_activations = torch.zeros(self.hidden_bias_weights.shape[0], self.hidden_bias_weights.shape[1])
    def forward(self, inputs):
        # print(inputs)
        # Inputs should be a tensor of shape [batch_size, num_inputs]
        # Apply input to hidden connections
        #+ torch.matmul(self.outputs, self.output_hidden_weights)
        hidden_activations = torch.matmul(inputs, self.input_hidden_weights) + torch.matmul(self.hidden_activations, self.hidden_recurrent_weights) # + self.hidden_bias_weights 
        # print(inputs)
        # print(hidden_activations)
        # print(self.input_hidden_weights)
        hidden_activations = torch.sigmoid(hidden_activations)  # Activation function
        self.hidden_activations = hidden_activations
        # Apply hidden to output connections
        # print(torch.matmul(self.outputs, self.output_recurrent_weights))
        outputs = torch.matmul(hidden_activations, self.hidden_output_weights) #+ torch.matmul(self.outputs, self.output_recurrent_weights) #+ self.output_bias_weights
        # print(outputs)
        s = torch.nn.Softmax(dim=-1)
        self.outputs = s(outputs)
        return self.outputs



def json_to_network_genome(data) -> NetworkGenome:
    network_genome = data["networkGenome"]
    node_genomes = list(map(lambda d: NodeGenome(d["id"], parse_node_type(d["type"]), parse_activation_function(d["activationFunction"]), d["bias"]), network_genome["nodeGenomes"]))
    connection_genes = list(map(lambda d: ConnectionGenome(d["id"], d["inputNode"], d["outputNode"], d["weight"], d["enabled"]), network_genome["connectionGenes"]))
    fitness = data.get('fitness')
    shared_fitness = data.get('sharedFitness')
    species_id = data.get('speciesId')
    return NetworkGenome(node_genomes, connection_genes, fitness, shared_fitness, species_id)

def json_to_network(json_data, network_builder: NetworkBuilder) -> Network:
    network_genome = json_to_network_genome(json_data)
    return network_builder.build_network_from_genome(network_genome)

def parse_activation_function(activation_function_name: str) -> ActivationFunction:
    return {
        "IDENTITY": ActivationFunction.IDENTITY,
        "SIGMOID": ActivationFunction.SIGMOID,
        "TANH": ActivationFunction.TANH,
        "RELU": ActivationFunction.RELU,
        "GAUSSIAN": ActivationFunction.GAUSSIAN,
        "SINE": ActivationFunction.SINE,
        "COS": ActivationFunction.COS,
        "ABS": ActivationFunction.ABS,
        "STEP": ActivationFunction.STEP
    }.get(activation_function_name, ActivationFunction.IDENTITY)

def parse_node_type(node_type_name: str) -> NodeType:
    return {
        "INPUT": NodeType.INPUT,
        "HIDDEN": NodeType.HIDDEN,
        "OUTPUT": NodeType.OUTPUT
    }.get(node_type_name, NodeType.INPUT)


class TaskNetwork2(torch.nn.Module):
    def __init__(self, substrate : Substrate, cppn_query : CPPNConnectionQuery):
        super(TaskNetwork2, self).__init__()
        self.substrate = substrate
        # self.self_attention = SelfAttention(weights_q=torch.nn.Parameter(torch.randn(substrate.hidden_coords[0].shape[0], substrate.hidden_coords[0].shape[1])), bias_q=torch.nn.Parameter(torch.randn(substrate.hidden_coords[0].shape[1])), weights_k=torch.nn.Parameter(torch.randn(substrate.hidden_coords[0].shape[0], substrate.hidden_coords[0].shape[1])), bias_k=torch.nn.Parameter(torch.randn(substrate.hidden_coords[0].shape[1])))
        # Get connection matrices with weights potentially being zero
        self.input_hidden_weights = substrate.get_layer_connections(substrate.input_coords, substrate.hidden_coords[0], cppn_query)
        self.hidden_hidden_weights = [substrate.get_layer_connections(substrate.hidden_coords[i], substrate.hidden_coords[i+1], cppn_query) for i in range(len(substrate.hidden_coords)-1)]
        self.hidden_output_weights = substrate.get_layer_connections(substrate.hidden_coords[-1], substrate.output_coords, cppn_query)
        # self.output_hidden_weights = substrate.get_layer_connections(substrate.output_coords, substrate.hidden_coords[0], cppn_query)
        self.hidden_bias_weights = [substrate.get_layer_connections(substrate.bias_coords, substrate.hidden_coords[i], cppn_query) for i in range(len(substrate.hidden_coords))]
        self.output_bias_weights = substrate.get_layer_connections(substrate.bias_coords, substrate.output_coords, cppn_query)
        self.hidden_recurrent_weights = [substrate.get_recurrent_layer_connections(substrate.hidden_coords[i], substrate.hidden_coords[i], cppn_query) for i in range(len(substrate.hidden_coords))]
        self.output_recurrent_weights = substrate.get_recurrent_layer_connections(substrate.output_coords, substrate.output_coords, cppn_query)
        self.outputs = torch.zeros(self.output_bias_weights.shape[0], self.output_bias_weights.shape[1])
        self.hidden_activations = [torch.zeros(self.hidden_bias_weights[i].shape[0], self.hidden_bias_weights[i].shape[1]) for i in range(len(self.hidden_bias_weights))]
    def forward(self, inputs):
        # print(inputs)
        # Inputs should be a tensor of shape [batch_size, num_inputs]
        # Apply input to hidden connections
        #+ torch.matmul(self.outputs, self.output_hidden_weights)
        # print(self.input_hidden_weights[self.input_hidden_weights > 1])
        # print(self.input_hidden_weights[self.input_hidden_weights < -1])
        # print(inputs)
        # print("====>")
        # print(self.hidden_activations[0])
        # print(self.hidden_recurrent_weights[0])
        # print( torch.matmul(self.hidden_activations[0], self.hidden_recurrent_weights[0]))
        # print("<====")
        # print(torch.matmul(inputs, self.input_hidden_weights))
        # exit(0)
        #+ torch.matmul(self.hidden_activations[0], self.hidden_recurrent_weights[0])
        self.hidden_activations[0] = torch.matmul(inputs, self.input_hidden_weights)  + self.hidden_bias_weights[0]#+ torch.matmul(self.hidden_activations[0], self.hidden_recurrent_weights[0])
        self.hidden_activations[0] = torch.sigmoid(self.hidden_activations[0])  # Activation function
        for i in range(len(self.substrate.hidden_coords)-1):
            # print("no loop")  + torch.matmul(self.hidden_activations[i+1], self.hidden_recurrent_weights[i+1]) 
            self.hidden_activations[i+1] = torch.matmul(self.hidden_activations[i], self.hidden_hidden_weights[i])+ self.hidden_bias_weights[i+1]  #+ torch.matmul(self.hidden_activations[i+1], self.hidden_recurrent_weights[i+1]) 
            self.hidden_activations[i+1] = torch.sigmoid(self.hidden_activations[i+1])  # Activation function
        # self.hidden_activations = hidden_activations
        # Apply hidden to output connections
        # print(torch.matmul(self.outputs, self.output_recurrent_weights))
        # print(self.input_hidden_weights)
        # print(self.hidden_activations[-1])
        # print(self.hidden_activations[0])
        outputs = torch.matmul(self.hidden_activations[-1], self.hidden_output_weights)+ self.output_bias_weights #+ torch.matmul(self.outputs, self.output_recurrent_weights) #+ self.output_bias_weights
        # print(outputs)
        # s = torch.nn.Softmax(dim=-1)
        self.outputs = outputs
        return self.outputs

