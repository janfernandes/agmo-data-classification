import time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, RocCurveDisplay

start_time = time.time()


def get_regra1():
    regra_1 = [[0, 0, 0] for i in range(GENOME_SIZE)]
    regra_1[19] = [0, 10, 1]
    regra_1[30] = [1, 10, 1]
    return regra_1


def get_regra2():
    regra_2 = [[0, 0, 0] for i in range(GENOME_SIZE)]
    regra_2[4] = [1, 10, 1]
    regra_2[26] = [1, 10, 1]
    regra_2[27] = [0, 10, 2]
    return regra_2


def get_regra3():
    regra_3 = [[0, 0, 0] for i in range(GENOME_SIZE)]
    regra_3[32] = [0, 10, 2]
    return regra_3


def get_regra4():
    regra_4 = [[0, 0, 0] for i in range(GENOME_SIZE)]
    regra_4[8] = [1, 10, 1]
    regra_4[10] = [1, 10, 1]
    regra_4[16] = [1, 10, 3]
    regra_4[24] = [1, 10, 2]
    regra_4[27] = [0, 10, 1]
    regra_4[31] = [0, 10, 1]
    return regra_4


def get_regra5():
    regra_5 = [[0, 0, 0] for i in range(GENOME_SIZE)]
    regra_5[11] = [1, 10, 1]
    regra_5[14] = [0, 10, 1]
    regra_5[23] = [1, 10, 1]
    return regra_5


def get_regra6():
    regra_6 = [[0, 0, 0] for i in range(GENOME_SIZE)]
    regra_6[6] = [0, 10, 1]
    regra_6[30] = [0, 10, 1]
    return regra_6


def create_representation():
    x, y = load_data('../bases/dermatology.data')
    return train_test_split(x, y, test_size=0.33, stratify=y, random_state=np.random.seed(11))


def aplicar_regra(X_train, y_train, regra):
    y_pred = []
    y_true = [1 if i == CLASSE else 0 for i in y_train]
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp) if (tn+fp) else 0
    sensitivity = tp / (tp+fn) if (tp+fn) else 0
    # specificity = recall_score(y_true, y_pred, average=None)[0]
    # sensitivity = recall_score(y_true, y_pred, average=None)[1]
    precision = tp / (tp+fp) if (tp+fp) else 0

    objetive_1 = sensitivity * specificity
    objetive_2 = (sensitivity * precision)/(sensitivity + precision) if sensitivity + precision else 0
    objetive_3 = (GENOME_SIZE - len(regra)+1) / GENOME_SIZE if len(regra) else 0

    # pesquisar o objetivo 4
    scores = np.r_[y_pred, y_true]
    y = np.r_[np.ones(len(y_pred)), np.zeros(len(y_true))]
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    # positives = list(y_train).count(CLASSE)
    # negatives = len(y_train) - positives
    # tpr = tp / positives
    # fpr = fp / negatives
    objetive_4 = metrics.auc(fpr, tpr)
    # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="teste").plot()
    # roc_display.plot()
    # plt.show()

    return objetive_4

def evaluate(X_train, y_train, pop):
    fit = []
    for individuo in pop:
        evaluate_ind(X_train, fit, individuo, y_train)
    return np.array(fit)


def evaluate_ind(X_train, fit, individuo, y_train):
    regra = []
    for j in range(GENOME_SIZE):
        if individuo[j][1] > LIMIAR:
            regra.append([j, individuo[j][0], individuo[j][2]])
    fit.append(aplicar_regra(X_train, y_train, regra))


def load_data(dataset):
    data = np.genfromtxt(dataset, delimiter=",")
    data = data[~np.isnan(data).any(axis=1)]
    # imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # imp.fit(data)
    # SimpleImputer()
    # data = imp.transform(data)
    return data[:, :-1], data[:, -1].astype(int)


def create_population():
    Pop = np.zeros((TP + CR, GENOME_SIZE, 3)).astype(int)
    Fit = np.zeros(TP + CR)

    # se categorico -> = e !=
    # se numerico -> >= e <
    # operators = ['>=', '<']
    # weight = np.asarray(np.arange(0, 1.1, 0.1))
    # se idade entao sao outros valores, se historico familiar sao outros valores
    # ler artigo p entender melhor
    # value = [0, 1, 2, 3]
    # family_history = [0, 1]

    Pop[:TP] = [generate_ind() for _ in range(TP)]

    return Pop, Fit


def generate_ind():
    idades = [0, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
              35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 60, 61,
              62, 63, 64, 65, 67, 68, 70, 75]
    # operador, peso e valor
    ind = [[np.random.randint(2), np.random.randint(11), np.random.randint(4)] for i in range(GENOME_SIZE)]
    ind[10][2] = np.random.randint(2)
    ind[33][2] = idades[np.random.randint(61)]
    return ind


def stochastic_tournament(Fit):
    max_v = sum(Fit)
    first_tour = np.random.choice(TP, CR, p=(None if max_v == 0 else Fit / max_v))
    second_tour = np.random.choice(TP, CR, p=(None if max_v == 0 else Fit / max_v))
    parents = [first_tour[i] if Fit[first_tour[i]] > Fit[second_tour[i]] else second_tour[i] for i
               in range(CR)]
    return parents


def two_points_crossover(parent1, parent2):
    gene1, gene2 = np.random.choice(GENOME_SIZE, 2, replace=False)
    new_child1 = np.copy(parent1)
    new_child2 = np.copy(parent2)
    aux = np.copy(parent1)
    if gene1 > gene2:
        gene1, gene2 = gene2, gene1

    new_child1[range(gene1, gene2)] = np.copy(new_child2[range(gene1, gene2)])
    new_child2[range(gene1, gene2)] = np.copy(aux[range(gene1, gene2)])
    return new_child1, new_child2


def mutate_gene_op(gene):
    return 1 if gene[0] == 0 else 0


def mutate_gene_weight():
    return np.random.randint(11)

def mutate_gene_value():
    return np.random.randint(4)


def ordered_reinsertion(Pop, fit):
    aux_pop = np.zeros(Pop.shape).astype(int)
    fit_sorted = np.argsort(-fit)[:TP]
    aux_pop[:TP] = Pop[fit_sorted]
    return aux_pop


def pure_reinsertion(Pop, fit):
    aux_pop = np.zeros(Pop.shape).astype(int)
    # apenas o melhor dos pais sobrevive
    melhor = np.copy(Pop[np.argmax(fit[:TP])])
    pior = np.argmin(fit[TP:])
    aux_pop[:TP] = np.copy(Pop[TP:])
    aux_pop[pior] = melhor
    return aux_pop


def extract_rule(popi):
    regra = []
    for j in range(GENOME_SIZE):
        if popi[j][1] > LIMIAR:
            regra.append([j, popi[j][0], popi[j][2]])
    return regra


def verify_ind(X_train, y_train):
    regra = get_regra6()
    fit = []
    evaluate_ind(X_train, fit, regra, y_train)
    print(fit)


def main():
    # crir a representacao
    np.random.seed(547)

    X_train, X_test, y_train, y_test = create_representation()
    # verify_ind(X_train, y_train)
    # verify_ind(X_test, y_test)

    for execution in range(EXECUTIONS):
        pop, fit = create_population()

        fit[:TP] = evaluate(X_train, y_train, pop[:TP])

        for generation in range(GEN):
            parents = stochastic_tournament(fit[:TP])

            for i in range(0, CR - 1, 2):
                pop[TP + i], pop[TP + i + 1] = two_points_crossover(pop[parents[i]], pop[parents[i + 1]])

            apply_mutation(pop)

            fit = evaluate(X_train, y_train, pop)
            pop = np.copy(pure_reinsertion(pop, fit))
            fit = evaluate(X_train, y_train, pop)

            print("O melhor encontrado na geracao %s" % (generation))
            print("Treino: %s" % (fit[np.argmax(fit[:TP])]))

            if fit[np.argmax(fit[:TP])] == 1:
                break

        print("Regra extraida")
        extracted_rules = extract_rule(pop[np.argmax(fit[:TP])])
        print(extracted_rules)
        print("teste: ", aplicar_regra(X_test, y_test, extracted_rules))

    #     if fit[np.argmax(fit[:TP])] == 1:
    #         cont += 1
    # print("Foram encontradas %s solucoes otimas em %s execucoes." % (cont, EXECUTIONS))
    print("--- %s segundos ---" % (time.time() - start_time))


def apply_mutation(pop):
    operator_mutation(pop)
    weight_mutation(pop)
    value_mutation(pop)


def operator_mutation(pop):
    mutations = np.random.randint(low=TP, high=TP + CR, size=int(TP * PMUT))
    for i in mutations:
        gene = np.random.choice(GENOME_SIZE, size=int(GENOME_SIZE * PMUT))
        for j in gene:
            pop[i][j][0] = mutate_gene_op(pop[i][j])

def weight_mutation(pop):
    mutations = np.random.randint(low=TP, high=TP + CR, size=int(TP * PMUT))
    for i in mutations:
        gene = np.random.choice(GENOME_SIZE, size=int(GENOME_SIZE * PMUT))
        for j in gene:
            pop[i][j][1] = mutate_gene_weight()

def value_mutation(pop):
    mutations = np.random.randint(low=TP, high=TP + CR, size=int(TP * PMUT))
    for i in mutations:
        gene = np.random.choice(GENOME_SIZE, size=int(GENOME_SIZE * PMUT))
        for j in gene:
            pop[i][j][2] = mutate_gene_value()


# constants
EXECUTIONS = 1
TP = 50
CR = 50  # quer dizer 100%
GEN = 50
PMUT = 0.3
TOURNAMENT_SIZE = 3
GENOME_SIZE = 34
LIMIAR = 7
CLASSE = 5

main()
