import kfp
from kfp import dsl
from kfp.components import func_to_container_op
import json


# Define a Kubeflow Pipeline component from a Python function
@func_to_container_op
def show_results(linear_regression, random_forest, g_boost) -> None:

    # Print results from different machine learning models
    print(f"linear regression: {linear_regression}")
    print(f"random forest: {random_forest}")
    print(f"gredient boost: {g_boost}")



# Define the main pipeline
@dsl.pipeline(name='Car price Pipeline', description='Definition of the main pipeline')
def car_price_pipeline():

    # Load yaml manifests for each component of the pipeline
    load = kfp.components.load_component_from_file('load_data/load_data.yaml')
    create_features = kfp.components.load_component_from_file('create_features/create_features.yaml')
    linear_regression = kfp.components.load_component_from_file('linear_regression/linear_regression.yaml')
    random_forest = kfp.components.load_component_from_file('random_forest/random_forest.yaml')
    g_boost = kfp.components.load_component_from_file('g_boost/g_boost.yaml')

    # Define the first task of the pipeline to load data
    load_task = load()

    # Define subsequent tasks in the pipeline that depend on the output of the load_task
    create_features_task = create_features(load_task.output)
    linear_regression_task = linear_regression(create_features_task.output)
    random_forest_task = random_forest(create_features_task.output)
    g_boost_task = g_boost(create_features_task.output)

    # Combine outputs from different models and show results
    show_results(linear_regression_task.output, random_forest_task.output, g_boost_task.output)

# Compile the pipeline into a yaml file when the script is run
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(car_price_pipeline, 'car_price_pipeline.yaml')