import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from datetime import timedelta
import functools

def timer(func):
    """Timer decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        func_val = func(*args, **kwargs)
        end = perf_counter()
        total_time = end - start
        return (func_val, total_time)
    return wrapper

def random_walks(q_vals, mean=0, std=0.01):
    for i in range(np.size(q_vals)):
        q_vals[i] += np.random.normal(mean, std)

def update_sample_average(q_val_hat, n_val, action, reward):
    n_val[action] += 1
    q_val_hat[action] += (reward - q_val_hat[action]) / n_val[action]

def update_constant_timestep(q_val_hat, param, action, reward):
    q_val_hat[action] += (reward - q_val_hat[action]) * param

def simulate_steps(num_steps, strats, p, q_vals, methods, action_dict, q_val_t, q_val_hat, best_action, param, n_val):
    action = {}
    for time_step in range(num_steps):
        random_walks(q_vals)
        strat = action_dict[np.random.choice(strats, p=p)]
        for method in methods:
            action = strat(q_val_hat[method])
            reward = np.random.normal(q_vals[action])
            q_val_t[method][time_step] = reward
            best_action[method][time_step] = int(action == np.argmax(q_vals))
        update_sample_average(q_val_hat["sample_average"], n_val, action, reward)
        update_constant_timestep(q_val_hat["constant_timestep"], param, action, reward)
        
@timer
def sim_k_bandits(q_vals=None, methods=("sample_average", "constant_timestep"), step_param=0.1, k=10, num_steps=10000, q_vals_std=0.1, epsilon=0.1):
    if q_vals is None:
        q_vals = np.zeros(k)
    n_val = np.zeros(k)
    q_val_hat, q_val_t, best_action = dict(), dict(), dict()
    for method in methods:
        q_val_hat[method] = np.zeros(k)
        q_val_t[method] = np.empty(num_steps)
        best_action[method] = np.zeros(num_steps)
        
    strats, p = ("explore", "exploit"), (epsilon, 1 - epsilon)
    action_dict = {
        "explore": lambda x: np.random.choice(x.size),
        "exploit": np.argmax
        }
    
    simulate_steps(num_steps, strats, p, q_vals, methods, action_dict, q_val_t, q_val_hat, best_action, step_param, n_val)
    
    return q_val_t, best_action

@timer
def run_tests(num_tests=1, k=10, num_steps=10000, methods=("sample_average", "constant_timestep")):
    
    def estimated_time(num_tests, curr_test, prev_avg, last_time):
        avg_time = prev_avg + (last_time - prev_avg) / curr_test
        ETA = avg_time * (num_tests - curr_test)
        output_string = f"\rTest {curr_test} of {num_tests} finished in "
        output_string += f"{last_time:.2f} seconds. ETA: {timedelta(seconds=round(ETA))} "
        print(output_string, end="")
        return avg_time

    print("Starting simulation:")
    q_val_t_all, best_action_all = dict(), dict()
    for method in methods:
        q_val_t_all[method] = np.zeros(num_steps)
        best_action_all[method] = np.zeros(num_steps)
        
    avg_time = 0
    for curr_test in range(1, num_tests + 1):
        ((q_val_t, best_action), last_time) = sim_k_bandits(q_vals=np.zeros(k), methods=methods, step_param=0.1, k=k, num_steps=num_steps, q_vals_std=0.1, epsilon=0.1)
        avg_time = estimated_time(num_tests, curr_test, avg_time, last_time)
        for method in methods:
            q_val_t_all[method] += q_val_t[method]
            best_action_all[method] += best_action[method]
    for method in methods:
        q_val_t_all[method] = q_val_t_all[method] / num_tests
        best_action_all[method] = best_action_all[method] / num_tests
        # best_action_all[method] = best_action_all[method].cumsum() / (np.arange(1, best_action_all[method].size + 1) * num_tests)
        
    return q_val_t_all, best_action_all
    
def plot_k_bandits(data, num_plots, methods, colors=("b", "g")):
    num_steps = q_val_t[methods[0]].size # Randomly choose a value in q_val_t to get num_steps

    fig, ax = plt.subplots(num_plots)
    fig.set_size_inches(10,7)
    ylabels = ('Average Reward', "% Optimal Actions")
    for i in range(num_plots):
        ax[i].set(xlabel="Steps", ylabel=ylabels[i])
        for method in methods:
            ax[i].plot(np.arange(1, num_steps + 1), data[i][method], color=colors[method], label=method)
        ax[i].legend()
       
    fig.savefig("ex_2-5.png", dpi=120)

# Execution starts here
num_tests = 2000
k = 10
num_steps = 30000
methods = ("sample_average", "constant_timestep")
((q_val_t, best_action), total_time) = run_tests(num_tests=num_tests, k=k, num_steps=num_steps, methods=methods)
print(f"\nTotal execution time: {timedelta(seconds=round(total_time))}")
colors = {methods[0]: "blue",
          methods[1]: "green"}
data = (q_val_t, best_action)
num_plots = len(data)
plot_k_bandits(data, num_plots, methods, colors)
