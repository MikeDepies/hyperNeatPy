from NetworkGenome import NodeType, ActivationFunction, NetworkGenome
from typing import Callable, List, Dict
from math import exp, tanh, sin, cos
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
            ActivationFunction.IDENTITY: lambda x: x,
            ActivationFunction.SIGMOID: lambda x: 1 / (1 + exp(-x)),
            ActivationFunction.TANH: lambda x: tanh(x),
            ActivationFunction.RELU: lambda x: max(0.0, x),
            ActivationFunction.GAUSSIAN: lambda x: exp(-x ** 2.0),
            ActivationFunction.SINE: lambda x: sin(x),
            ActivationFunction.COS: lambda x: cos(x),
            ActivationFunction.ABS: lambda x: abs(x),
            ActivationFunction.STEP: lambda x: 0.0 if x < 0 else 1.0
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