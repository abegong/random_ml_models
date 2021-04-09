import random
import pandas as pd
from numpy import (
    corrcoef,
    var,
    random,
)

from random_ml_models import (
    generate_random_ml_model,
    split_data_and_train_model,
    LogitModelMonitoringProfiler,
)

def test_smoke():
    result = generate_random_ml_model(
        num_variables=5,
        num_rows=200,
    )

    assert result.keys() == set(["variables", "sample_data"])

    assert len(result["variables"]) == 5
    for variable in result["variables"]:
        assert set(variable.keys()) == set([
            'variable_type',
            "link_function",
            "link_function_params",
            'noise_distribution',
            'noise_distribution_params',
        ])
        print(variable)

    # assert False

    D = result["sample_data"]
    assert D.shape == (200, 5)

    df = pd.DataFrame(D)
    print(df.head(10))


    print(var(D, axis=1))
    # print( corrcoef(D) )
    print("-"*80)
    print( corrcoef(D)[-1,:] )
    # print( corrcoef(x=D[0,:], y=D[1,:])[0,1] )

def test_split_data_and_train_model():
    data = pd.DataFrame({
        "x1": random.uniform(size=100),
        "x2": random.uniform(size=100),
        "y": random.uniform(size=100),
    }).to_numpy()
    result = split_data_and_train_model(data=data)
    assert set(result.keys()) == set([
        'X_test',
        'Y_test',
        'Y_hat_test',
        'X_train',
        'Y_train',
        'Y_hat_train',
        'model',
    ])

def test_():
    data_for_model = generate_random_ml_model(
        num_variables=5,
        num_rows=200,
    )["sample_data"]

    trained_model_and_data = split_data_and_train_model(
        data=data_for_model
    )

    my_profiler = LogitModelMonitoringProfiler()
    expectations, validation_results = my_profiler.profile(
        # TODO: append yhat_train to this array
        trained_model_and_data["X_train"]
    )
