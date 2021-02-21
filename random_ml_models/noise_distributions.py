import numpy as np


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

