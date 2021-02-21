import random
import numpy as np

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

LINK_FUNCTIONS = {
    "FLOAT" : {
        "linear" : {
            "gen_params_func" : generate_linear_float_params,
            "gen_sample_data_func" : generate_linear_float_variable,
        }
    },
}
