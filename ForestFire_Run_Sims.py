import numpy as np
from random import uniform
import matplotlib.pyplot as plt
import pandas as pd
import csv 
import os

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

from forest_fire.agent import TreeCell
from forest_fire.model import ForestFire

from multiprocessing import Pool

def generateSims(j):
    density = uniform(0.2, 1)

    fire = ForestFire(100, 100, density)
    while fire.count_type(fire, "On Fire") > 0:
        fire.step()
    results = fire.datacollector.get_model_vars_dataframe()

    params = []
    params.append(list([density]))

    df_inputs = []
    df_inputs.append(list(results['Fine']))

    df_inputs2 = []
    df_inputs2.append(list(results['On Fire']))

    df_outputs = []
    df_outputs.append(list(results['Burned Out']))

    for i in range(99):
        density = uniform(0.2, 1)

        fire = ForestFire(100, 100, density)
        while fire.count_type(fire, "On Fire") > 0:
            fire.step()
        results = fire.datacollector.get_model_vars_dataframe()
        params.append(list([density]))
        df_inputs.append(list(results['Fine']))
        df_inputs2.append(list(results['On Fire']))
        df_outputs.append(list(results['Burned Out']))

    params_file_name = "train_inputs_" + str(j) + ".csv"    
    input_file_name = "train_outputs_fine_" + str(j) + ".csv"
    input2_file_name = "train_outputs_burning_" + str(j) + ".csv"
    output_file_name = "train_outputs_burnedout_" + str(j) + ".csv"

    os.chdir('/Users/maria/Desktop/WSC_simulations/forest_fire_data')

    with open(params_file_name, "w") as f:
        f.truncate()
        wr = csv.writer(f)
        wr.writerows(params)
        f.close()

    with open(input_file_name, "w") as f:
        f.truncate()
        wr = csv.writer(f)
        wr.writerows(df_inputs)
        f.close()

    with open(input2_file_name, "w") as f:
        f.truncate()
        wr = csv.writer(f)
        wr.writerows(df_inputs2)
        f.close()

    with open(output_file_name, "w") as f:
        f.truncate()
        wr = csv.writer(f)
        wr.writerows(df_outputs)
        f.close()
    
def run_generate_sims(operation, input, pool):
    pool.map(operation, input)
    
if __name__ == '__main__':
    processes_count = 8
    processes_pool = Pool(processes_count)
    os.chdir('/Users/maria/Desktop/Schelling_ML/Forest_fire_sims')
    run_generate_sims(generateSims, range(100), processes_pool)   

