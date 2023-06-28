"""
This is a boilerplate pipeline 'training_lstm'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (pytorch_lstm_init, pytorch_lstm_init, pytorch_lstm_fit, pytorch_lstm_predict, pytorch_lstm_save)
from .nodes import (check_prediction_output_mse, draw_prediction_and_target, pytorch_lstm_load, fetch_model_from_mlflow_and_predict, check_mse_is_valid_set_model_to_prduction)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=pytorch_lstm_init,
            #     inputs=["params:pytorch_lstm_init_options"],
            #     outputs="pytorch_lstm_init_model",
            #     name="pytorch_lstm_init"
            # ),
            node(
                func=pytorch_lstm_fit,
                inputs=[
                    "apple_data_target_column_numpy_array_scaled_training_input_pytorch_tensor",
                    "apple_data_target_column_numpy_array_scaled_training_target_pytorch_tensor",
                    "params:pytorch_lstm_fit_options"
                ],
                outputs=[
                        "pytorch_lstm_after_fit",
                        "mlflow_run_id",
                        "mlflow_model_uri",
                        "mlflow_model_name",
                        "mlflow_model_version",
                    ],
                name="pytorch_lstm_fit"
            ),
            # node(
            #     func=pytorch_lstm_save,
            #     inputs=[
            #         "pytorch_lstm_after_fit",
            #         "params:pytorch_lstm_save_options"
            #     ],
            #     outputs="pytorch_lstm_save_path",
            #     name="pytorch_lstm_save_after_fit"
            # ),
            #
            # node(
            #     func=pytorch_lstm_load,
            #     inputs=[
            #         "pytorch_lstm_save_path",
            #         "params:pytorch_lstm_load_options"
            #     ],
            #     outputs="pytorch_lstm_fit_model_from_load",
            #     name="pytorch_lstm_load_fit_model"
            # ),
            node(
                func=pytorch_lstm_predict,
                inputs=[
                    "pytorch_lstm_after_fit",
                    "apple_data_target_column_numpy_array_scaled_testing_input_pytorch_tensor",
                    "scaler"
                ],
                outputs="apple_data_target_column_numpy_array_scaled_prediction_output_pytorch_tensor",
                name="pytorch_lstm_predict_output"
            ),
            node(
                func=check_prediction_output_mse,
                inputs=[
                    "apple_data_target_column_numpy_array_scaled_prediction_output_pytorch_tensor",
                    "apple_data_target_column_numpy_array_scaled_testing_target_pytorch_tensor",
                    "mlflow_run_id",
                ],
                outputs="lstm_prediction_output_mse",
                name="lstm_prediction_output_mse_calculation"
            ),
            node(
                func=draw_prediction_and_target,
                inputs=[
                    "apple_data_target_column_numpy_array_scaled_prediction_output_pytorch_tensor",
                    "apple_data_target_column_numpy_array_scaled_testing_target_pytorch_tensor"
                ],
                outputs=None,
                name="draw_prediction_and_target_plot"
            ),
            node(
                func=fetch_model_from_mlflow_and_predict,
                inputs=[
                    "mlflow_model_uri",
                    "apple_data_target_column_numpy_array_scaled_testing_input_pytorch_tensor",
                    "scaler"
                ],
                outputs="apple_data_target_column_numpy_array_scaled_prediction_output_pytorch_tensor_from_mlflow_check",
                name="fetch_model_from_mlflow_and_predict_cross_check"
            ),
            node(
                func=check_prediction_output_mse,
                inputs=[
                    "apple_data_target_column_numpy_array_scaled_prediction_output_pytorch_tensor_from_mlflow_check",
                    "apple_data_target_column_numpy_array_scaled_testing_target_pytorch_tensor",
                    "mlflow_run_id",
                ],
                outputs="lstm_prediction_output_mse_cross_check",
                name="lstm_prediction_output_mse_calculation_cross_check"
            ),
            node(
                func=check_mse_is_valid_set_model_to_prduction,
                inputs=[
                    "mlflow_model_name",
                    "mlflow_model_version",
                    "lstm_prediction_output_mse_cross_check"
                ],
                outputs=None,
                name="check_mse_is_valid_transit_model_stage_to_prduction"
            ),
        ]
    )
    # return pipeline(
    #     [
    #         node(
    #             func=extracting_training_data,
    #             inputs=["apple_data", "params:extracting_options"],
    #             outputs="apple_data_target_column_numpy_array",
    #             name="extracting_training_data_from_csv_to_numpy_array"
    #         ),
    #         node(
    #             func=scaling_by_column,
    #             inputs=["apple_data_target_column_numpy_array"],
    #             outputs="apple_data_target_column_numpy_array_scaled",
    #             name="scaling_numpy_array"
    #         ),
    #         node(
    #             func=splitting,
    #             inputs=[
    #                 "apple_data_target_column_numpy_array_scaled",
    #                 "params:splitting_options"
    #             ],
    #             outputs=[
    #                 "apple_data_target_column_numpy_array_scaled_training",
    #                 "apple_data_target_column_numpy_array_scaled_testing"
    #             ],
    #             name="splitting_numpy_array"
    #         ),
    #         node(
    #             func=sliding_window_masking,
    #             inputs=[
    #                 "apple_data_target_column_numpy_array_scaled_training",
    #                 "params:sliding_window_options"
    #             ],
    #             outputs=[
    #                 "apple_data_target_column_numpy_array_scaled_training_input",
    #                 "apple_data_target_column_numpy_array_scaled_training_target"
    #             ],
    #             name="sliding_window_masking_training"
    #         ),
    #         node(
    #             func=convert_data_to_pytorch_tensor,
    #             inputs=[
    #                 "apple_data_target_column_numpy_array_scaled_training_input",
    #                 "apple_data_target_column_numpy_array_scaled_training_target"
    #             ],
    #             outputs=[
    #                 "apple_data_target_column_numpy_array_scaled_training_input_pytorch_tensor",
    #                 "apple_data_target_column_numpy_array_scaled_training_target_pytorch_tensor"
    #             ],
    #             name="converting_training_data_to_pytorch_tensor"
    #         ),
    #     ]
    # )



