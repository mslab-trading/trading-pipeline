from strategies.run_gino import *
from strategies.run_allen import *
from strategies.utils.analysis import print_result
from deap import base, creator, tools, algorithms
import random
import contextlib
import io

def get_score(returns: pd.Series, flag):
    pct = returns.pct_change().dropna()
    if (flag == "roi"):
        return (returns.iloc[-1] / returns.iloc[0]) ** (240/returns.shape[0]) - 1
    elif (flag == "sharpe"):
        return pct.mean() / pct.std() * np.sqrt(240)
    else:
        return (returns.iloc[-1] / returns.iloc[0]) ** (240/returns.shape[0]) - 1

def ga_optimizer(function, result_dir, parameter_names, lower_bounds, upper_bounds, flag="roi", population_size=30, num_epochs=10):
    num_param = len(parameter_names)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    def create_individual():
        return [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(num_param)]

    toolbox.register("individual", lambda: creator.Individual(create_individual()))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(ind):
        f = open("config/backtest.yaml")
        cfg_dict_backtest = yaml.safe_load(f)
        f = open("config/main_training.yaml")
        cfg_dict_train = yaml.safe_load(f)
        cfg = cfg_dict_backtest | cfg_dict_train
        for i in range (num_param):
            cfg[parameter_names[i]] = ind[i]
        with contextlib.redirect_stdout(io.StringIO()):
            result = function(cfg, result_dir)
        score = get_score(result['model'].returns, flag)
        print(f"param:{ind}, score:{score}")
        return (score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    toolbox.register("mutate", tools.mutGaussian, mu=[0 for i in range (len(parameter_names))], sigma=[(upper_bounds[i]-lower_bounds[i])/10 for i in range (len(parameter_names))], indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)
    '''
    population = toolbox.population(n=population_size)
    result_pop, _ = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2,
                                        ngen=num_epochs, verbose=False)

    # 找出最佳個體
    best = tools.selBest(result_pop, k=1)[0]
    return best, f(*best)
    fnwofewf
    ''' 
    population = toolbox.population(n=population_size)
    max_fitnesses = []
    avg_fitnesses = []

    for gen in range(num_epochs):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        for ind in offspring:
            for i in range (num_param):
                ind[i] = float(np.clip(ind[i], lower_bounds[i], upper_bounds[i]))

        # 評估新個體
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # 選出新一代
        population = toolbox.select(offspring, k=len(population))
        # 紀錄統計資訊
        fits = [ind.fitness.values[0] for ind in population]
        max_fitnesses.append(np.max(fits))
        avg_fitnesses.append(np.mean(fits))
        
        best_ind = max(population, key=lambda ind: ind.fitness.values[0])
        print(f"Gen {gen}: Max = {max_fitnesses[-1]:.4f}, Avg = {avg_fitnesses[-1]:.4f}, Param = {best_ind}")
    print(f"best param: {best_ind}")
    
if __name__ == "__main__":
    ga_optimizer(get_allen_result, "results/Example_Result", ["ADX_threshold", "buy_percentile", "sell_percentile"], [0,0,0], [50,1,1])
    #ga_optimizer(get_gino_result, "results/Example_Result", ["max_positions", "max_holding_period"], [1,1], [100,100])