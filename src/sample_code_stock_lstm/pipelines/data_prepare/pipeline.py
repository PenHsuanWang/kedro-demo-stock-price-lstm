"""
This is a boilerplate pipeline 'data_prepare'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (extracting_training_data, scaling_by_column, splitting, sliding_window_masking, convert_data_to_pytorch_tensor)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extracting_training_data,
                inputs=["apple_data", "params:extracting_options"],
                outputs="apple_data_target_column_numpy_array",
                name="extracting_training_data_from_csv_to_numpy_array"
            ),
            node(
                func=scaling_by_column,
                inputs=["apple_data_target_column_numpy_array"],
                outputs=[
                        "apple_data_target_column_numpy_array_scaled",
                        "scaler"
                    ],
                name="scaling_numpy_array"
            ),
            node(
                func=splitting,
                inputs=[
                    "apple_data_target_column_numpy_array_scaled",
                    "params:splitting_options"
                ],
                outputs=[
                    "apple_data_target_column_numpy_array_scaled_training",
                    "apple_data_target_column_numpy_array_scaled_testing"
                ],
                name="splitting_numpy_array"
            ),
            node(
                func=sliding_window_masking,
                inputs=[
                    "apple_data_target_column_numpy_array_scaled_training",
                    "params:sliding_window_options"
                ],
                outputs=[
                    "apple_data_target_column_numpy_array_scaled_training_input",
                    "apple_data_target_column_numpy_array_scaled_training_target"
                ],
                name="sliding_window_masking_training"
            ),
            node(
                func=convert_data_to_pytorch_tensor,
                inputs=[
                    "apple_data_target_column_numpy_array_scaled_training_input",
                    "apple_data_target_column_numpy_array_scaled_training_target"
                ],
                outputs=[
                    "apple_data_target_column_numpy_array_scaled_training_input_pytorch_tensor",
                    "apple_data_target_column_numpy_array_scaled_training_target_pytorch_tensor"
                ],
                name="converting_training_data_to_pytorch_tensor"
            ),
            node(
                func=sliding_window_masking, # prepare for testing dataset
                inputs=[
                    "apple_data_target_column_numpy_array_scaled_testing",
                    "params:sliding_window_options"
                ],
                outputs=[
                    "apple_data_target_column_numpy_array_scaled_testing_input",
                    "apple_data_target_column_numpy_array_scaled_testing_target"
                ],
                name="sliding_window_masking_testing"
            ),
            node(
                func=convert_data_to_pytorch_tensor, # prepare for testing dataset
                inputs=[
                    "apple_data_target_column_numpy_array_scaled_testing_input",
                    "apple_data_target_column_numpy_array_scaled_testing_target"
                ],
                outputs=[
                    "apple_data_target_column_numpy_array_scaled_testing_input_pytorch_tensor",
                    "apple_data_target_column_numpy_array_scaled_testing_target_pytorch_tensor"
                ],
                name="converting_testing_data_to_pytorch_tensor"
            ),
        ]
    )
