import random
import numpy as np

def generate_linear_float_params(
    variables,
    num_rows,
):
    k = len(variables)
    return {
        "weights": np.asarray([[random.choice([-1,1])*random.lognormvariate(-2, .15) for i in range(k)]]),
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

def generate_linear_int_params(
    variables,
    num_rows,
):
    return generate_linear_float_params(
        variables,
        num_rows,
    )

def generate_linear_int_variable(
    sample_data,
    noise,
    weights,
):
    return generate_linear_float_variable(
        sample_data,
        noise,
        weights,
    ).astype(int)

def generate_logit_params(
    variables,
    num_rows,
):
    return generate_linear_float_params(
        variables,
        num_rows,
    )

def generate_logit_variable(
    sample_data,
    noise,
    weights,
):
    num_vars = weights.shape[1]
    if num_vars > 0:
        x = sample_data[:num_vars]
        x = x/np.std(x)
        new_sample_data = np.dot(weights, x)
        #Set the mean to zero before adding noise
        new_sample_data -= new_sample_data.mean()
        new_sample_data += noise

    else:
        new_sample_data = noise

    return new_sample_data > 0


LINK_FUNCTIONS = {
    "FLOAT" : {
        "linear" : {
            "gen_params_func" : generate_linear_float_params,
            "gen_sample_data_func" : generate_linear_float_variable,
        }
    },
    "INT" : {
        "linear" : {
            "gen_params_func" : generate_linear_int_params,
            "gen_sample_data_func" : generate_linear_int_variable,
        }
    },
    "BINARY" : {
        "logit" : {
            "gen_params_func" : generate_logit_params,
            "gen_sample_data_func" : generate_logit_variable,
        }
    },
}
