import random
import numpy as np
from typing import List, Dict

from .link_functions import *
from .noise_distributions import NOISE_DISTRIBUTIONS

VARIABLE_TYPES = [
    "INT",
    "FLOAT",
    "BINARY",
    # "CATEGORICAL",
]

def generate_random_variable(
    previous_variables : List,
    num_variables : int,
    num_rows: int,
):
    variable_type = random.choice(VARIABLE_TYPES)

    link_function = random.choice(list(LINK_FUNCTIONS[variable_type].keys()))
    link_function_params = LINK_FUNCTIONS[variable_type][link_function]["gen_params_func"](
        previous_variables,
        num_rows,
    )

    noise_distribution = random.choice(list(NOISE_DISTRIBUTIONS[variable_type].keys()))
    noise_distribution_params = NOISE_DISTRIBUTIONS[variable_type][noise_distribution]["gen_params_func"](
        size=num_rows,        
    )

    return {
        "variable_type" : variable_type,
        "link_function" : link_function,
        "link_function_params" : link_function_params,
        "noise_distribution" : noise_distribution,
        "noise_distribution_params" : noise_distribution_params,
    }

def generate_sample_data(
    sample_data,
    variable_type,
    link_function,
    link_function_params,
    noise_distribution,
    noise_distribution_params,
):
    noise_distribution_meta = NOISE_DISTRIBUTIONS[variable_type][noise_distribution]
    noise_distribution_func = noise_distribution_meta["gen_sample_data_func"]
    noise = noise_distribution_func(**noise_distribution_params)

    # print(link_function_params)
    link_function_meta = LINK_FUNCTIONS[variable_type][link_function]
    link_function_func = link_function_meta["gen_sample_data_func"]
    new_sample_data = link_function_func(
        sample_data=sample_data,
        noise=noise,
        **link_function_params
    )

    if np.isnan(new_sample_data).any():
        print("="*80)
        print(variable_type)
        print(link_function)
        print(link_function_params)
        print(noise_distribution)
        print(noise_distribution_params)
        print(new_sample_data)
        print("-"*80)
        assert False

    # print(new_sample_data)

    return new_sample_data

def generate_random_ml_model(
    num_variables : int=3,
    num_rows: int=20,
):

    variables = []
    sample_data = np.zeros((num_variables, num_rows))

    for i in range(num_variables):
        new_variable = generate_random_variable(
            variables,
            # sample_data,
            num_variables,
            num_rows,
        )
        variables.append(new_variable)

        sample_data[i,:] = generate_sample_data(
            sample_data,
            **new_variable
        )

    return {
        "variables" : variables,
        "sample_data" : sample_data
    }
