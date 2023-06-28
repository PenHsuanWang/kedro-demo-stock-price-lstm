"""
This is a boilerplate pipeline 'training_lstm'
generated using Kedro 0.18.10
"""

import time

import os

import torch
import torch.nn as nn

import numpy as np

import mlflow.pytorch
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def pytorch_lstm_init(params) -> torch.nn.Module:

    model = LSTMModel(params['input_size'], params['hidden_size'], params['output_size'])

    return model


def pytorch_lstm_fit(train_data, train_target, params):
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    current_time = round(time.time())
    mlflow.set_experiment("/kedro-demo-pytorch-{}".format(current_time))
    
    model = LSTMModel(
            input_size=train_data.shape[2],
            hidden_size=params['hidden_size'],
            output_size=params['output_size']
        )
    
    
    with mlflow.start_run(run_name='pytorch-lstm-training'):
        mlflow.pytorch.autolog()
        
        mlflow.log_param("input_size", train_data.shape[2])
        mlflow.log_param("hidden_size", params['hidden_size'])
        mlflow.log_param("output_size", params['output_size'])
        mlflow.log_param("num_epochs", params['num_epochs'])
        mlflow.log_param("first_layer", type(model.lstm1))
        mlflow.log_param("second_layer", type(model.lstm2))
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        for epoch in range(params['num_epochs']):
            outputs = model(train_data)
            optimizer.zero_grad()
            loss = criterion(outputs, train_target)
            mlflow.log_metric("loss", loss.item())
            loss.backward()
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))
        
        
        # regist the model
        model_name = "pytorch-lstm-model"
        artifact_path = "model"
        run_id = mlflow.active_run().info.run_id
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        
        # log model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            # signature=signature,
            registered_model_name=model_name,
        )
        
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(model_details)
        
        from mlflow.tracking.client import MlflowClient
        from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
        
        client = MlflowClient(tracking_uri='http://ec2-44-213-176-187.compute-1.amazonaws.com:7005')
        
        def wait_until_ready(model_name, model_version):
            
            for _ in range(10):
                model_version_details = client.get_model_version(
                name=model_name,
                version=model_version,
                )
                status = ModelVersionStatus.from_string(model_version_details.status)
                print("Model status: %s" % ModelVersionStatus.to_string(status))
                if status == ModelVersionStatus.READY:
                    break
                time.sleep(1)
        
        wait_until_ready(model_details.name, model_details.version)
        
        return model, run_id, model_uri, model_details.name, model_details.version


def pytorch_lstm_predict(model, test_data: np.ndarray, scaler) -> np.ndarray:

    # model.eval()
    with torch.no_grad():
        output = model(test_data)
        output = output.cpu().numpy()

    # output = scaler.inverse_transform(output)

    return output

def fetch_model_from_mlflow_and_predict(mlflow_model_uri, test_data: np.ndarray, scaler) -> np.ndarray:
    
    # setup the mlflow tracking server by mlflow client
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    # fetch the model
    model = mlflow.pytorch.load_model(model_uri=mlflow_model_uri)
    
    with torch.no_grad():
        output = model(test_data)
        output = output.cpu().numpy()

    # output = scaler.inverse_transform(output)

    return output


def pytorch_lstm_save(model, params) -> str:

    torch.save(model.state_dict(), f"{params['model_save_path']}")

    return f"{params['model_save_path']}"


def pytorch_lstm_load(load_path: str, params) -> torch.nn.Module:
    # load model checkpoint from file
    model = torch.load(f"{params['model_load_path']}")
    return model


def check_prediction_output_mse(prediction: np.ndarray, target: torch.Tensor, mlflow_run_id):
    # Calculation of the mean squared error between the prediction and the target   
    
    target = target.cpu().numpy()
    mse = np.mean((prediction - target) ** 2)
    print(f"Mean squared error: {mse:.2f}")
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metric("mse", mse)
    return mse

def check_mse_is_valid_set_model_to_prduction(model_name, model_version, mse):
    mlflow_client = MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])
    
    if mse < 0.5:
        mlflow_client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )
        
    return None

def draw_prediction_and_target(prediction: np.ndarray, target: np.ndarray):
    # Plotting the prediction and the target
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(target, label="target")
    plt.plot(prediction, label="prediction")
    plt.legend()
    plt.show()


