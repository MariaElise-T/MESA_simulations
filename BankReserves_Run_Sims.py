import itertools
import csv
import os

import mesa
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd

from bank_reserves.agents import Bank, Person
from bank_reserves.model import BankReserves

# Start of datacollector functions

from multiprocessing import Pool

def generateSims(j):
    reserve = np.random.uniform(0, 1)

    model = BankReserves(20, 20, init_people=2, rich_threshold=10, reserve_percent = 50)
    for i in range(100):
        model.step()
    results = model.datacollector.get_model_vars_dataframe()

    params = []
    params.append(list([reserve]))

    df_inputs = []
    df_inputs.append(list(results['Rich']))

    df_inputs2 = []
    df_inputs2.append(list(results['Middle Class']))

    df_outputs = []
    df_outputs.append(list(results['Poor']))

    for i in range(99):
        reserve = np.random.uniform(0, 1)

        model = BankReserves(20, 20, init_people=2, rich_threshold=10, reserve_percent = 50)
        for i in range(100):
            model.step()
        results = model.datacollector.get_model_vars_dataframe()
        params.append(list([reserve]))
        df_inputs.append(list(results['Rich']))
        df_inputs2.append(list(results['Middle Class']))
        df_outputs.append(list(results['Poor']))

    params_file_name = "train_inputs_" + str(j) + ".csv"    
    input_file_name = "train_outputs_rich_" + str(j) + ".csv"
    input2_file_name = "train_outputs_middleclass_" + str(j) + ".csv"
    output_file_name = "train_outputs_poor_" + str(j) + ".csv"

    os.chdir('/Users/maria/Desktop/WSC_simulations/bank_reserves_data')

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
    #os.chdir('/Users/maria/Desktop/Schelling_ML/Forest_fire_sims')
    run_generate_sims(generateSims, range(100), processes_pool)   