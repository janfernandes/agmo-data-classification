import collections

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.crossover import Crossover
from pymoo.model.duplicate import ElementwiseDuplicateElimination
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

CLASSE = 4  # aqui sao 2 e 4
LIMIAR = 7


class DataMiningProblem(Problem):

    def __init__(self, n_obj=2, n_genes=9):
        super().__init__(n_var=1, n_obj=n_obj, n_constr=0, elementwise_evaluation=True)
        self.n_characters = n_genes
        self.OPERATOR = [0, 1]
        self.WEIGHT = np.asarray(np.arange(0, 11, 1))
        self.VALUE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.LIMIAR = 7
        self.CLASSE = CLASSE
        self.PMUT = 0.3
        self.TP = 50
        self.CR = 50

    def _evaluate(self, individuo_rep, out, *args, **kwargs):
        individuo = individuo_rep[0]

        regra = []
        for j in range(self.n_characters):
            # limiar
            if individuo[j][1] > self.LIMIAR:
                regra.append([j, individuo[j][0], individuo[j][2]])
        results_matrix, y_pred, y_true = self.aplicar_regra(regra)
        tn, fp, fn, tp = results_matrix
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0

        objetivo_1 = sensitivity * specificity
        objetivo_2 = (sensitivity * precision) / (sensitivity + precision) if sensitivity + precision else 0
        objetivo_3 = (self.n_characters - len(regra) + 1) / self.n_characters if len(regra) else 0
        # pesquisar o objetivo 4
        scores = np.r_[y_pred, y_true]
        y = np.r_[np.ones(len(y_pred)), np.zeros(len(y_true))]
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        objetivo_4 = metrics.auc(fpr, tpr)

        out["F"] = np.array([- objetivo_1, - objetivo_2, - objetivo_3], dtype=np.float)

    def aplicar_regra(self, regra):
        y_pred = []
        y_true = [1 if i == self.CLASSE else 0 for i in y_train]
        eh_doente = 0
        for paciente in X_train:
            for item in regra:
                eh_doente = 0
                # '>='
                if item[1] == 0:
                    if paciente[item[0]] >= item[2]:
                        eh_doente = 1
                    else:
                        break
                # '<'
                else:
                    if paciente[item[0]] < item[2]:
                        eh_doente = 1
                    else:
                        break
            y_pred.append(eh_doente)
        return confusion_matrix(y_true, y_pred).ravel(), y_pred, y_true


class DataMiningSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=np.object)

        for i in range(n_samples):
            X[i, 0] = [
                [np.random.choice(problem.OPERATOR), np.random.choice(problem.WEIGHT), np.random.choice(problem.VALUE)]
                for _ in range(problem.n_characters)]

        return X


class DataMiningCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2, prob=1)

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=np.object)

        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            parent1, parent2 = X[0, k, 0], X[1, k, 0]

            gene1, gene2 = np.random.choice(problem.n_characters, 2, replace=False)
            # prepare the offsprings
            new_child1 = np.copy(parent1)
            new_child2 = np.copy(parent2)

            aux = np.copy(parent1)
            if gene1 > gene2:
                gene1, gene2 = gene2, gene1

            new_child1[range(gene1, gene2)] = np.copy(new_child2[range(gene1, gene2)])
            new_child2[range(gene1, gene2)] = np.copy(aux[range(gene1, gene2)])

            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = new_child1, new_child2

        return Y


class DataMiningMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        self.operator_mutation(X, problem)
        self.weight_mutation(X, problem)
        self.value_mutation(X, problem)

        return X

    def operator_mutation(self, X, problem):
        mutations = np.random.randint(low=0, high=problem.TP, size=int(problem.TP * problem.PMUT))
        for i in mutations:
            gene = np.random.choice(problem.n_characters, size=int(problem.n_characters * problem.PMUT))
            for j in gene:
                X[i, 0][j][0] = self.mutate_gene_op(X[i, 0][j][0])

    def mutate_gene_op(self, gene):
        return 1 if gene == 0 else 0

    def weight_mutation(self, X, problem):
        mutations = np.random.randint(low=0, high=problem.TP, size=int(problem.TP * problem.PMUT))
        for i in mutations:
            gene = np.random.choice(problem.n_characters, size=int(problem.n_characters * problem.PMUT))
            for j in gene:
                X[i, 0][j][1] = self.mutate_gene_weight()

    def mutate_gene_weight(self):
        return np.random.randint(11)

    def value_mutation(self, X, problem):
        mutations = np.random.randint(low=0, high=problem.TP, size=int(problem.TP * problem.PMUT))
        for i in mutations:
            gene = np.random.choice(problem.n_characters, size=int(problem.n_characters * problem.PMUT))
            for j in gene:
                X[i, 0][j][2] = self.mutate_gene_value()

    def mutate_gene_value(self):
        return np.random.randint(11)


class DataMiningDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return collections.Counter(str(a.X[0])) == collections.Counter(str(b.X[0]))


def load_data(dataset):
    data = np.genfromtxt(dataset, delimiter=",")
    data = data[~np.isnan(data).any(axis=1)]
    return data[:, 1:-1].astype(int), data[:, -1].astype(int)


def create_representation():
    x, y = load_data('../bases/breast-cancer-wisconsin.data')
    return train_test_split(x, y, test_size=0.33, stratify=y, random_state=np.random.seed(11))


X_train, X_test, y_train, y_test = create_representation()

algorithm_NSGA2 = NSGA2(pop_size=50,
                        sampling=DataMiningSampling(),
                        crossover=DataMiningCrossover(),
                        mutation=DataMiningMutation(),
                        eliminate_duplicates=False)

res = minimize(DataMiningProblem(n_obj=4),
               algorithm_NSGA2,
               ('n_gen', 50),
               seed=10,
               verbose=False)


# Scatter().add(res.F).show()
# print("Resultados em treino:", res.F[0] * -1)
# results = res.X[np.argsort(res.F[:, 0])]
# print(np.column_stack([results]))

# print(np.unique(-1*np.column_stack([res.F])))
# print("tam: ", len(np.unique(-1 * np.column_stack([res.F]))))
# print("------------")


def plotar_quatro_objetivos():
    input_data = -1 * np.column_stack([res.F])
    x = input_data[:, 0]
    y = input_data[:, 1]
    z = input_data[:, 2]
    v = input_data[:, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(x, y, z, edgecolor="gray", color="None")
    ax.scatter(x, y, z, c=v, s=100)
    ax.set_xlabel('Objetivo 1')
    ax.set_ylabel('Objetivo 2')
    ax.set_zlabel('Objetivo 3')
    fig.colorbar(surf, shrink=0.8, aspect=70, label='Objetivo 4', orientation="horizontal", pad=0.05)
    plt.savefig('C:\\Users\\ferna\\OneDrive\\Imagens\\gina\\result-cancer-mama-1-2-3-4-classe-' + str(CLASSE) + '.png')
    plt.show()


# plotar_quatro_objetivos()


def plotar_dois_objetivos():
    scores = -1 * np.column_stack([res.F])

    def identify_pareto(scores):
        # Count number of items
        population_size = scores.shape[0]
        # Create a NumPy index for scores on the pareto front (zero indexed)
        population_ids = np.arange(population_size)
        # Create a starting list of items on the Pareto front
        # All items start off as being labelled as on the Parteo front
        pareto_front = np.ones(population_size, dtype=bool)
        # Loop through each item. This will then be compared with all other items
        for i in range(population_size):
            # Loop through all other items
            for j in range(population_size):
                # Check if our 'i' pint is dominated by out 'j' point
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
        # Return ids of scenarios on pareto front
        return population_ids[pareto_front]

    pareto = identify_pareto(scores)
    print('Pareto front index vales')
    print('Points on Pareto front: \n', len(pareto))

    pareto_front = scores[pareto]
    print('\nPareto front scores')
    print(pareto_front)

    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values

    x_all = scores[:, 0]
    y_all = scores[:, 1]
    x_pareto = pareto_front[:, 0]
    y_pareto = pareto_front[:, 1]

    plt.scatter(x_all, y_all, color='xkcd:dusty lavender')
    plt.plot(x_pareto, y_pareto, color='xkcd:purple red')
    plt.xlabel('Objetivo 1')
    plt.ylabel('Objetivo 4')
    plt.savefig('C:\\Users\\ferna\\OneDrive\\Imagens\\gina\\result-cancer-mama-1-4-classe-' + str(CLASSE) + '.png')
    plt.show()


def plot_tres_objetivos():
    scores = -1 * np.column_stack([res.F])
    X = scores[:, 0]
    Y = scores[:, 1]
    Z = scores[:, 2]

    ax = plt.axes(projection='3d')  # Data for a three-dimensional line
    # ax.plot3D(X, Y, Z, color='none')  # Data for three-dimensional scattered points
    ax.scatter(X, Y, Z, color='xkcd:purple red')

    # 1. create vertices from points
    verts = [list(zip(X, Y, Z))]
    # 2. create 3d polygons and specify parameters
    srf = Poly3DCollection(verts, alpha=.2, facecolor='xkcd:purple red')
    # 3. add polygon to the figure (current axes)
    plt.gca().add_collection3d(srf)
    ax.set_xlabel('Objetivo 1')
    ax.set_ylabel('Objetivo 2')
    ax.set_zlabel('Objetivo 3')

    plt.savefig('C:\\Users\\ferna\\OneDrive\\Imagens\\gina\\result-cancer-mama-1-2-3-classe-' + str(CLASSE) + '.png')
    plt.show()


plot_tres_objetivos()
# plotar_tres_objetivos()
# print(-1*np.column_stack([res.F])[:, 0])
#
# print("------------")
# print("SOLUCOES")
# print("------------")
# print(-1*np.column_stack([res.F])[:, 1])

# print("------------")
# print("SOLUCOES")
# print("------------")
# print(-1*np.column_stack([res.F])[:, 2])


# def extract_rule(popi):
#     regra = []
#     for j in range(34):
#         if popi[j][1] > LIMIAR:
#             regra.append([j, popi[j][0], popi[j][2]])
#     return regra
#
#
# extracted_rules = extract_rule(results[0][0])
#
#
# def aplicar_regra(X_train, y_train, regra):
#     y_pred = []
#     y_true = [1 if i == CLASSE else 0 for i in y_train]
#     eh_doente = 0
#     for paciente in X_train:
#         for item in regra:
#             eh_doente = 0
#             # '>='
#             if item[1] == 0:
#                 if paciente[item[0]] >= item[2]:
#                     eh_doente = 1
#                 else:
#                     break
#             # '<'
#             else:
#                 if paciente[item[0]] < item[2]:
#                     eh_doente = 1
#                 else:
#                     break
#         y_pred.append(eh_doente)
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     specificity = tn / (tn + fp) if (tn + fp) else 0
#     sensitivity = tp / (tp + fn) if (tp + fn) else 0
#
#     precision = tp / (tp + fp) if (tp + fp) else 0
#
#     objetivo_1 = sensitivity * specificity
#     objetivo_2 = (sensitivity * precision) / (sensitivity + precision) if sensitivity + precision else 0
#     objetivo_3 = (34 - len(regra) + 1) / 34 if len(regra) else 0
#     # pesquisar o objetivo 4
#     scores = np.r_[y_pred, y_true]
#     y = np.r_[np.ones(len(y_pred)), np.zeros(len(y_true))]
#     fpr, tpr, thresholds = metrics.roc_curve(y, scores)
#     objetivo_4 = metrics.auc(fpr, tpr)
#
#     return objetivo_1, objetivo_3
#
#
# print("Regra extraida")
# print(extracted_rules)
#
# print("************************************************************************")
# print("Resultado em teste: ", aplicar_regra(X_test, y_test, extracted_rules))
