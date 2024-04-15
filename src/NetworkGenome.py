# Define the NodeGenome class
from enum import Enum
class NodeGenome:
    def __init__(self, id, type, activation_function, bias):
        self.id = id
        self.type = type
        self.activation_function = activation_function
        self.bias = bias

# Define the ConnectionGenome class
class ConnectionGenome:
    def __init__(self, id, input_node, output_node, weight, enabled):
        self.id = id
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        self.enabled = enabled

# Define the NetworkGenome class
class NetworkGenome:
    def __init__(self, node_genomes, connection_genes, fitness=None, shared_fitness=None, species_id=None):
        self.node_genomes = node_genomes
        self.connection_genes = connection_genes
        self.fitness = fitness
        self.shared_fitness = shared_fitness
        self.species_id = species_id

    def render(self):
        builder = "Network Genome:\n"
        builder += "Nodes:\n"
        for node in self.node_genomes:
            builder += f" - Node ID: {node.id}, Type: {node.type}, Activation Function: {node.activation_function}, Bias: {node.bias}\n"
        builder += "Connections:\n"
        for connection in self.connection_genes:
            builder += f" - Connection ID: {connection.id}, Input Node ID: {connection.input_node}, Output Node ID: {connection.output_node}, Weight: {connection.weight}, Enabled: {connection.enabled}\n"
        if self.fitness:
            builder += f"Fitness: {self.fitness}\n"
        if self.shared_fitness:
            builder += f"Shared Fitness: {self.shared_fitness}\n"
        if self.species_id:
            builder += f"Species ID: {self.species_id}\n"
        return builder

class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class ActivationFunction(Enum):
    IDENTITY = 1
    SIGMOID = 2
    TANH = 3
    RELU = 4
    GAUSSIAN = 5
    SINE = 6
    COS = 7
    ABS = 8
    STEP = 9

    @classmethod
    def cppn(cls):
        return [cls.IDENTITY, cls.SIGMOID, cls.TANH, cls.RELU, cls.GAUSSIAN, cls.SINE, cls.COS, cls.ABS, cls.STEP]
