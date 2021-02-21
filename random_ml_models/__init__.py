import random
import numpy as np
from typing import List, Dict


def generate_linear_float_params(
    variables,
    num_rows,
):
    k = len(variables)
    return {
        "weights": np.asarray([[random.normalvariate(0, 1) for i in range(k)]]),
    }

def generate_linear_float_variable(
    sample_data,
    noise,
    weights,
):
    num_vars = weights.shape[1]
    if num_vars > 0:
        new_sample_data = np.dot(weights, sample_data[:num_vars])
        new_sample_data += noise
    else:
        new_sample_data = noise

    return new_sample_data


VARIABLE_TYPES = [
    # "INT",
    "FLOAT",
    # "BINARY",
    # "CATEGORICAL",
]

LINK_FUNCTIONS = {
    "FLOAT" : {
        "linear" : {
            "gen_params_func" : generate_linear_float_params,
            "gen_sample_data_func" : generate_linear_float_variable,
        }
    },
}

NOISE_DISTRIBUTIONS = {
    "FLOAT" : {
        "normal" : {
            "gen_params_func" : lambda size: { "loc": 0, "scale": 1, "size": size },
            "gen_sample_data_func" : np.random.normal,
        },
        "uniform" : {
            "gen_params_func" : lambda size: { "low": 0, "high": 1, "size": size },
            "gen_sample_data_func" : np.random.uniform,
        }
    }
}


def generate_random_variable(
    previous_variables : List,
    # previous_sample_data: np.ndarray,
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
