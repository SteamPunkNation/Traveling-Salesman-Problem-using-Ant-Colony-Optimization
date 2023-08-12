# Andrew Donate
# 7/3/23 - 7/9/23
# Project 3

# Based off of the following
# https://github.com/rochakgupta/aco-tsp/blob/master/aco_tsp.py

import math
import random as rand
import matplotlib.pyplot as plt

# For print colors
RED = '\033[91m'
RESET = '\033[0m'


class SolveTSPUsingACO:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.distance = 0.0

        def _select_node(self):
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            random_value = rand.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            self.tour = [rand.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, colony_size=10, min_scaling_factor=0.001, alpha=1.0, beta=3.0,
                rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None, labels=None):
        self.generation_distances = []
        self.colony_size = colony_size
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                x_diff = self.nodes[i][0] - self.nodes[j][0]
                y_diff = self.nodes[i][1] - self.nodes[j][1]
                distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, distance, initial_pheromone)
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

        # Store the initial tour
        self.initial_tour = [i for i in range(self.num_nodes)]


    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def _max_min(self):
        print('Started : MinMax')
        for step in range(self.steps):
            iteration_best_tour = None
            iteration_best_distance = float("inf")
            for ant in self.ants:
                ant.find_tour()
                if ant.get_distance() < iteration_best_distance:
                    iteration_best_tour = ant.tour
                    iteration_best_distance = ant.distance
            if float(step + 1) / float(self.steps) <= 0.75:
                self._add_pheromone(iteration_best_tour, iteration_best_distance)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_distance
            else:
                if iteration_best_distance < self.global_best_distance:
                    self.global_best_tour = iteration_best_tour
                    self.global_best_distance = iteration_best_distance
                self._add_pheromone(self.global_best_tour, self.global_best_distance)
                max_pheromone = self.pheromone_deposit_weight / self.global_best_distance
            min_pheromone = max_pheromone * self.min_scaling_factor
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)
                    if self.edges[i][j].pheromone > max_pheromone:
                        self.edges[i][j].pheromone = max_pheromone
                    elif self.edges[i][j].pheromone < min_pheromone:
                        self.edges[i][j].pheromone = min_pheromone

            self.generation_distances.append(iteration_best_distance)
            print("Current best distance for generation " + str(step + 1) + ": " + str(iteration_best_distance))
        
        print('Ended : MinMax')
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))

    def plot(self, line_width=2, point_radius=math.sqrt(5.0), annotation_size=8):
        x = []
        y = []

        for i, node_index in enumerate(self.global_best_tour):
            x.append(self.nodes[node_index][0])
            y.append(self.nodes[node_index][1])

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})
        axs[0].set_aspect('equal')
        axs[0].plot(x, y, linewidth=line_width)
        axs[0].plot([x[0], x[-1]], [y[0], y[-1]], linestyle='dotted', color='red')
        axs[0].scatter(x, y, s=math.pi * (point_radius ** 2.0), color='gold')
        for i, txt in enumerate(self.labels):
            axs[0].annotate(txt, (self.nodes[i][0], self.nodes[i][1]), size=annotation_size)
        axs[0].set_title('ACO MinMax')
        legend_elements = [
            plt.Line2D([0], [0], marker='o', markersize=point_radius, color='green', label='Start'),
            plt.Line2D([0], [0], marker='o', markersize=point_radius, color='red', label='End'),
            plt.Line2D([0], [0], marker='o', markersize=point_radius, color='gold', label='Other')
        ]
        axs[0].legend(handles=legend_elements)

        generations = range(1, len(self.generation_distances) + 1)
        axs[1].set_aspect('equal')
        axs[1].plot(generations, self.generation_distances, linewidth=line_width)
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Distance')
        axs[1].set_title('Generational Change: Distance vs. Generation')

        initial_x = [self.nodes[i][0] for i in self.initial_tour]
        initial_y = [self.nodes[i][1] for i in self.initial_tour]
        axs[2].set_aspect('equal')
        axs[2].plot(initial_x, initial_y, linewidth=line_width)
        axs[2].plot([initial_x[0], initial_x[-1]], [initial_y[0], initial_y[-1]], linestyle='dotted', color='red')
        axs[2].scatter(initial_x, initial_y, s=math.pi * (point_radius ** 2.0), color='blue')
        for i, txt in enumerate(self.labels):
            axs[2].annotate(txt, (self.nodes[i][0], self.nodes[i][1]), size=annotation_size)
        axs[2].set_title('Initial Path')

        plt.tight_layout()
        plt.show()


def userSettings():

    print("Parameters for ACO")

    # Number of cities
    numberOfCities = input(
        "Enter the integer number of cities in TSP (default is 25): ")
    if not checkUserInputInt(numberOfCities):
        print(RED + "Not valid number, defaulting to 25." + RESET)
        numberOfCities = 25

    # Population Size
    populationSize = input(
        "Enter the integer number of population size in TSP (default is 100): ")
    if not checkUserInputInt(populationSize):
        print(RED + "Not valid number, defaulting to 25." + RESET)
        populationSize = 100

    # Number of generations
    numberOfGenerations = input(
        "Enter the integer number of generations in TSP (default is 100): ")
    if not checkUserInputInt(numberOfGenerations):
        print(RED + "Not valid number, defaulting to 25." + RESET)
        numberOfGenerations = 100
    
    numberOfCities = int(numberOfCities)
    populationSize = int(populationSize)
    numberOfGenerations = int(numberOfGenerations)

    cities = [(rand.random() * 200, rand.random() * 200) for _ in range(0, numberOfCities)]
    max_min = SolveTSPUsingACO(colony_size=populationSize, steps=numberOfGenerations, nodes=cities)
    max_min._max_min()
    max_min.plot()

def checkUserInputInt(input):
    try:
        return int(input)
    except ValueError:
        return False


def checkUserInputFloat(input):
    try:
        return float(input)
    except ValueError:
        return False


# MAIN
if __name__ == '__main__':
    userSettings()
