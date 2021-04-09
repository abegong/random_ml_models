import great_expectations as ge

class LogitModelMonitoringProfiler():
    """Takes a dataframe of the shape X | y_hat and returns Expectations suitable for monitoring future data and predictions in the same format

    Note: we're not basing this class on the existing great_expectations Profiler classes, since they are going to change soon.
    """

    def profile(cls, data):
        df = ge.from_pandas(data)
        
        evr = df.expect_table_columns_to_match_ordered_list([])
        columns = evr["result"]["observed_value"]

        cardinalities = {}

        #for each column
        for column in columns:
            # TODO: base this on an actual Expectation
            cardinalities[column] = "numeric"

            # expect_column_values_to_not_be_null
            # expect_column_values_to_be_of_type
            # expect_column_kl_divergence_to_be_less_than

            if cardinalities[column] == "categorical":
                # expect_column_values_to_be_in_set
                # expect_column_most_common_value_to_be_in_set
                # expect_column_unique_value_count_to_be_between
                # expect_column_distinct_values_to_equal_set
                # expect_column_distinct_values_to_be_in_set
                pass

            if cardinalities[column] == "numeric":
                # expect_column_values_to_be_between
                # expect_column_mean_to_be_between
                # expect_column_median_to_be_between
                # expect_column_quantile_values_to_be_between
                # expect_column_stdev_to_be_between
                # expect_column_max_to_be_between
                # expect_column_min_to_be_between
                pass

        # for each column pair
        for column_A in columns:
            for column_B in columns:

                if column_A == column_B:
                    continue

                # for numeric-numeric column pairs:
                if cardinalities[column_A] == "numeric" and cardinalities[column_B] == "numeric":
                    # expect_column_pair_cramers_phi_value_to_be_less_than
                    pass

                if cardinalities[column_A] == "categorical" and cardinalities[column_B] == "numeric":
                    pass

                if cardinalities[column_A] == "numeric" and cardinalities[column_B] == "categorical":
                    pass

                if cardinalities[column_A] == "categorical" and cardinalities[column_B] == "categorical":
                    # expect_column_pair_cramers_phi_value_to_be_less_than
                    pass

        # TODO: return an actual ExpectationSuite and ValidationResults
        return None, None
