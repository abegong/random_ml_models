import random
import numpy as np

def generate_normal_float_params(size):
    loc = random.choice([
        0, 1, random.randint(-10, 10), random.uniform(-100, 100)
    ])
    scale = random.choice([
        1, random.randint(1, 10), random.uniform(0, 100), np.abs(loc)*random.uniform(0,10)
    ])

    return {
        "loc": loc,
        "scale": scale,
        "size": size
    }

def generate_uniform_float_params(size):
    low = random.choice([
        0, 1, random.randint(-10, 10), random.uniform(-100, 100)
    ])
    high = low + random.choice([
        1, random.randint(0, 10), random.uniform(0, 100)
    ])

    return {
        "low": low,
        "high": high,
        "size": size
    }

def generate_beta_float_params(size):
    a = 1/random.random()
    b = 1/random.random()

    return {
        "a": a,
        "b": b,
        "size": size
    }

def generate_normal_int_params(size):
    loc = random.choice([
        0, 1, random.randint(-10, 10), random.uniform(-100, 100)
    ])
    scale = random.choice([
        1, random.randint(1, 10), random.uniform(0, 100), np.abs(loc)*random.uniform(0,10)
    ])

    return {
        "loc": loc,
        "scale": scale,
        "size": size
    }

def generate_uniform_int_params(size):
    low = random.choice([
        0, 1, random.randint(-10, 10), random.randint(-100, 100)
    ])
    high = low + random.choice([
        1, random.randint(1, 10), random.randint(1, 100)
    ])

    return {
        "low": low,
        "high": high,
        "size": size
    }

def generate_normal_binary_params(size):
    loc = random.choice([
        random.uniform(-2, 2)
    ])

    return {
        "loc": loc,
        "scale": 1,
        "size": size
    }

def generate_beta_k_categorical_params(size):
    a = 1/random.random()
    b = 1/random.random()
    k = random.choice([
        random.randint(3, 5),
        random.randint(3, 10),
        random.randint(3, 20),
    ])

    return {
        "a": a,
        "b": b,
        "k": k,
        "size": size
    }

def generate_beta_k_categorical_noise(a, b, k, size):
    return (np.random.beta(a, b, size) * k).astype(int)

NOISE_DISTRIBUTIONS = {
    "FLOAT" : {
        "normal" : {
            "gen_params_func" : generate_normal_float_params,
            "gen_sample_data_func" : np.random.normal,
        },
        "uniform" : {
            "gen_params_func" : generate_uniform_float_params,
            "gen_sample_data_func" : np.random.uniform,
        }
    },
    "INT" : {
        "normal" : {
            "gen_params_func" : generate_normal_int_params,
            "gen_sample_data_func" : np.random.normal,
        },
        "uniform" : {
            "gen_params_func" : generate_uniform_int_params,
            "gen_sample_data_func" : np.random.uniform,
        }
    },
    "BINARY" : {
        "normal" : {
            "gen_params_func" : generate_normal_binary_params,
            "gen_sample_data_func" : np.random.normal,
        },
    },
    "CATEGORICAL" : {
        "beta_k" : {
            "gen_params_func" : generate_beta_k_categorical_params,
            "gen_sample_data_func" : generate_beta_k_categorical_noise,
        },
    }
}

