import random
import numpy as np

def generate_linear_float_params(
    variables,
    num_rows,
):
    weights = []
    for var in variables:
        if var["variable_type"] == "CATEGORICAL":
            new_weight = random.choice([-1,1])*random.lognormvariate(-2, .15)

            # #Instead of a single parameter, add a different parameter for each value of the categorical variable, 
            # k = var["noise_distribution_params"]["k"]
            # new_weight = [random.choice([-1,1])*random.lognormvariate(-2, .15) for j in range(k)]
        else:
            new_weight = random.choice([-1,1])*random.lognormvariate(-2, .15)

        weights.append(new_weight)

    return {
        "weights": np.asarray([weights]),
    }

def generate_linear_float_variable(
    sample_data,
    noise,
    weights,
):
    num_vars = weights.shape[1]
    if num_vars > 0:
        #Normalize the SD of all variables
        x = sample_data[:num_vars]
        x = x/np.std(x)

        new_sample_data = np.dot(weights, x)
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
        #Normalize the SD of all variables
        x = sample_data[:num_vars]
        x = x/np.std(x)
        new_sample_data = np.dot(weights, x)

        #Set the mean to zero before adding noise
        new_sample_data -= new_sample_data.mean()
        new_sample_data += noise

    else:
        new_sample_data = noise

    return new_sample_data > 0

def generate_categorical_params(
    variables,
    num_rows,
):
    return {
        "weights" : None
    }

def generate_categorical_variable(
    sample_data,
    noise,
    weights,
):
    return noise


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
    "CATEGORICAL" : {
        "categorical" : {
            "gen_params_func" : generate_categorical_params,
            "gen_sample_data_func" : generate_categorical_variable,
        }
    },
}
