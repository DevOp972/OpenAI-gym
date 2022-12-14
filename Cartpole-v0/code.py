# IMPORTS
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

# from test import mutation

env = gym.make('CartPole-v0')


class NeuralNet:
    """
    Neural network to optimize the cartpole environment
    """

    def __init__(self, input_dim, hidden_dim, output_dim, test_run):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.test_run = test_run

    # helper functions
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def init_weights(self):
        input_weight = []
        input_bias = []

        hidden_weight = []
        out_weight = []

        input_nodes = 4

        for i in range(self.test_run):
            inp_w = np.random.rand(self.input_dim, input_nodes)
            input_weight.append(inp_w)
            inp_b = np.random.rand((input_nodes))
            input_bias.append(inp_b)
            hid_w = np.random.rand(input_nodes, self.hidden_dim)
            hidden_weight.append(hid_w)
            out_w = np.random.rand(self.hidden_dim, self.output_dim)
            out_weight.append(out_w)

        return [input_weight, input_bias, hidden_weight, out_weight]

    def forward_prop(self, obs, input_w, input_b, hidden_w, out_w):

        obs = obs/max(np.max(np.linalg.norm(obs)), 1)
        Ain = self.relu(obs@input_w + input_b.T)
        Ahid = self.relu(Ain@hidden_w)
        Zout = Ahid @ out_w
        A_out = self.relu(Zout)
        output = self.softmax(A_out)

        return np.argmax(output)

    def run_environment(self, input_w, input_b, hidden_w, out_w):
        obs = env.reset()
        score = 0
        time_steps = 300
        for i in range(time_steps):
            action = self.forward_prop(obs, input_w, input_b, hidden_w, out_w)
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        return score

    def run_test(self):
        generation = self.init_weights()
        input_w, input_b, hidden_w, out_w = generation
        scores = []
        for ep in range(self.test_run):
            score = self.run_environment(
                input_w[ep], input_b[ep], hidden_w[ep], out_w[ep])
            scores.append(score)
        return [generation, scores]


class GA:
    """
    Training neural net using genetic algorithm
    """

    def __init__(self, init_weight_list, init_fitness_list, number_of_generation, pop_size, learner, mutation_rate=0.5):
        # initilize different parameters of the GA
        self.number_of_generation = number_of_generation
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.current_generation = init_weight_list
        self.current_fitness = init_fitness_list
        self.best_gen = []
        self.best_fitness = -1000
        self.fitness_list = []
        self.learner = learner

    def crossover(self, DNA_list):
        """
        Generate number of offsprings from parents in DNA_list such that pop_size remains same.
        Think of an optimal crossover strategy
        """
        newDNAs = []
        eta = 0.6

        # 1.select 2 random dna from list
        # 2.crossover using differential selection
        # 3.iterate for pop_size-dna_list size

        while len(newDNAs) < (self.population_size-2):
            u = random.random()
            if u < 0.5:
                b = pow(2*u, 1/(eta+1))
            else:
                b = pow(1/(2*(1-u)), 1/(eta+1))

            p1 = np.array(DNA_list[0])
            p2 = np.array(DNA_list[1])

            c1 = 0.5*((1+b)*p1+(1-b)*p2)
            c2 = 0.5*((1-b)*p1+(1+b)*p2)
            newDNAs.append(c1)
            newDNAs.append(c2)
        if(len(newDNAs) > (self.population_size-2)):
            newDNAs.pop()
        # crossover
        # return newDNAs
        return newDNAs

    def mutation(self, DNA):
        """
        Mutate DNA. Use mutation_rate to determine the mutation probability.
        Make changes in the DNA.
        """
        dna = np.array(DNA)
        mutants = np.empty((dna.shape))
        # mutation_rate = self.mutation_rate
        mutation_rate = 0.4
        for i in range(mutants.shape[0]):
            random_value = random.random()
            mutants[i, :] = dna[i, :]
            if random_value > mutation_rate:
                continue
            int_random_value = random.randint(0, dna.shape[1]-1)
            # mutants[i, int_random_value] += np.random.uniform(-1.0, 1.0, 1)
            # print(np.std(mutants[i]))
            mutants[i,
                    int_random_value] += np.random.normal(0, np.std(mutants[i]))

        return mutants

    def next_generation(self):
        """
        Forms next generation from current generation.
        Before writing this function think of an appropriate representation of an individual in the population.
        Suggested method: Convert it into a 1-D array/list. This conversion is done for you in this function. Feel free to use any other method.
        Steps
        1. Crossover
        Suggested Method: select top two individuals with max fitness. generate remaining offsprings using these two individuals only.
        2. Mutation:
        """
        index_good_fitness = (-np.array(self.current_fitness)).argsort()[:2]
        # index of parents selected for crossover.
        # fill the list.

        new_DNA_list = []
        new_fitness_list = []

        DNA_list = []
        for index in index_good_fitness:
            w1 = self.current_generation[0][index]
            dna_in_w = w1.reshape(w1.shape[1], -1)

            b1 = self.current_generation[1][index]
            dna_b1 = np.append(dna_in_w, b1)

            w2 = self.current_generation[2][index]
            dna_whid = w2.reshape(w2.shape[1], -1)
            dna_w2 = np.append(dna_b1, dna_whid)

            wh = self.current_generation[3][index]
            dna = np.append(dna_w2, wh)
            DNA_list.append(dna)

        # parents selected for crossover moves to next generation
        new_DNA_list += DNA_list

        new_DNA_list += self.crossover(DNA_list)
        new_DNA_list = self.mutation(new_DNA_list)

        # mutate the new_DNA_list

        # converting 1D representation of individual back to original (required for forward pass of neural network)
        new_input_weight = []
        new_input_bias = []
        new_hidden_weight = []
        new_output_weight = []

        for newdna in new_DNA_list:

            newdna_in_w1 = np.array(
                newdna[:self.current_generation[0][0].size])
            new_in_w = np.reshape(
                newdna_in_w1, (-1, self.current_generation[0][0].shape[1]))
            new_input_weight.append(new_in_w)

            new_in_b = np.array(
                [newdna[newdna_in_w1.size:newdna_in_w1.size+self.current_generation[1][0].size]]).T  # bias
            new_input_bias.append(new_in_b)

            sh = newdna_in_w1.size + new_in_b.size
            newdna_in_w2 = np.array(
                [newdna[sh:sh+self.current_generation[2][0].size]])
            new_hid_w = np.reshape(
                newdna_in_w2, (-1, self.current_generation[2][0].shape[1]))
            new_hidden_weight.append(new_hid_w)

            sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
            new_out_w = np.array([newdna[sl:]]).T
            new_out_w = np.reshape(
                new_out_w, (-1, self.current_generation[3][0].shape[1]))
            new_output_weight.append(new_out_w)

            # evaluate fitness of new individual and add to new_fitness_list.
            # check run_environment function for details.
        for i in range(self.population_size):
            new_fitness_list.append(self.learner.run_environment(
                new_input_weight[i], new_input_bias[i], new_hidden_weight[i], new_output_weight[i]))

        new_generation = [new_input_weight, new_input_bias,
                          new_hidden_weight, new_output_weight]

        return new_generation, new_fitness_list

    def show_fitness_graph(self):
        """
        Show the fitness graph
        Use fitness_list to plot the graph
        """
        plt.plot(list(range(self.number_of_generation)),
                 self.fitness_list, label='Max Fitness')
        plt.legend()
        plt.title('Fitness through the generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.show()
        # plot

    def evolve(self):
        """
        Evolve the population
        Steps
        1. Iterate for number_of_generation and generate new population
        2. Find maximum fitness of an individual in this generation and update best_fitness 
        3. Append max_fitness to fitness_list
        4. Plot the fitness graph at end. Use show_fitnes_graph()
        """

        for i in range(self.number_of_generation):
            self.current_generation, self.current_fitness = self.next_generation()
            self.fitness_list.append(max(self.current_fitness))
            if(self.best_fitness <= self.fitness_list[i]):
                self.best_gen = self.current_generation
                self.best_fitness = self.fitness_list[i]
                index = self.current_fitness.index(self.best_fitness)
        # evolve
        self.show_fitness_graph()
        return self.best_gen, self.best_fitness, index


def trainer():
    pop_size = 15
    num_of_generation = 100
    learner = NeuralNet(
        env.observation_space.shape[0], 2, env.action_space.n, pop_size)
    init_weight_list, init_fitness_list = learner.run_test()
    # instantiate the GA optimizer
    ga = GA(init_weight_list, init_fitness_list, num_of_generation,
            pop_size, learner, mutation_rate=0.5)
    best_gen, best_fitness, index = ga.evolve()
    # call evolve function to obtain optimized weights.

    input_wt = best_gen[0][index]
    input_bias = best_gen[1][index]
    hidden_wt = best_gen[2][index]
    output_wt = best_gen[3][index]
    # return optimized weights

    return [input_wt, input_bias, hidden_wt, output_wt]


def test_run_env(params):
    input_w, input_b, hidden_w, out_w = params
    obs = env.reset()
    score = 0
    learner = NeuralNet(
        env.observation_space.shape[0], 2, env.action_space.n, 15)
    for t in range(5000):
        env.render()
        action = learner.forward_prop(obs, input_w, input_b, hidden_w, out_w)
        obs, reward, done, info = env.step(action)
        score += reward
        print(f"time: {t}, fitness: {score}")
        if done:
            break
    print(f"Final score: {score}")


def main():
    params = trainer()
    # print(np.array(params).shape)

    test_run_env(params)


if(__name__ == "__main__"):
    main()
