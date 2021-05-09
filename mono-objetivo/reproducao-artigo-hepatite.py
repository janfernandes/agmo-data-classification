import time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

start_time = time.time()


def create_representation():
    x, y = load_data('../bases/hepatitis.data')
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
    specificity = tn / (tn + fp) if (tn + fp) else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    # specificity = recall_score(y_true, y_pred, average=None)[0]
    # sensitivity = recall_score(y_true, y_pred, average=None)[1]
    precision = tp / (tp + fp) if (tp + fp) else 0

    objetive_1 = sensitivity * specificity
    objetive_2 = (sensitivity * precision) / (sensitivity + precision) if sensitivity + precision else 0
    objetive_3 = (GENOME_SIZE - len(regra) + 1) / GENOME_SIZE if len(regra) else 0

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

    return objetive_1


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
    return data[:, 1:-1], data[:, 0].astype(int)


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
    # operador, peso e valor
    ind = [[np.random.randint(2), np.random.randint(11), np.random.randint(2)] for _ in range(GENOME_SIZE)]
    ind[0][2] = np.random.choice(age, 1)[0]
    ind[13][2] = np.random.choice(bilirulin, 1)[0]
    ind[14][2] = np.random.choice(alk, 1)[0]
    ind[15][2] = np.random.choice(sgot, 1)[0]
    ind[16][2] = np.random.choice(albumin, 1)[0]
    ind[17][2] = np.random.choice(prostime, 1)[0]
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


def mutate_gene_value(gene):
    if gene == 1:
        return np.random.choice(age, 1)[0]
    if gene == 14:
        return np.random.choice(bilirulin, 1)[0]
    if gene == 15:
        return np.random.choice(alk, 1)[0]
    if gene == 16:
        return np.random.choice(sgot, 1)[0]
    if gene == 17:
        return np.random.choice(albumin, 1)[0]
    if gene == 18:
        return np.random.choice(prostime, 1)[0]
    return np.random.randint(2)


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
            pop[i][j][2] = mutate_gene_value(j)


age = [10, 20, 30, 40, 50, 60, 70, 80]
bilirulin = [0.39, 0.80, 1.20, 2.00, 3.00, 4.00]
alk = [33, 80, 120, 160, 200, 250]
sgot = [13, 100, 200, 300, 400, 500]
albumin = [2.1, 3.0, 3.8, 4.5, 5.0, 6.0]
prostime = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# constants
EXECUTIONS = 1
TP = 50
CR = 50  # quer dizer 100%
GEN = 50
PMUT = 0.3
TOURNAMENT_SIZE = 3
GENOME_SIZE = 18
LIMIAR = 7
CLASSE = 2  # neste caso sao as 1 e a 2

main()
