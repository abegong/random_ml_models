import random

from random_ml_models import (
    generate_random_ml_model
)
from numpy import corrcoef

def test_smoke():
    result = generate_random_ml_model(
        num_variables=5,
        num_rows=200,
    )

    print("="*80)

    assert result.keys() == set(["variables", "sample_data"])

    # assert len(result["variables"]) == 3
    for variable in result["variables"]:
        assert set(variable.keys()) == set([
            'variable_type',
            "link_function",
            "link_function_params",
            'noise_distribution',
            'noise_distribution_params',
        ])
        # print(variable)

    D = result["sample_data"]
    print( D )
    # assert D.shape == (3, 50)
    # print("*"*80)
    print( corrcoef(D) )
    print( corrcoef(x=D[0,:], y=D[1,:])[0,1] )

    assert False