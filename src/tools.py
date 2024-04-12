import os, json
from langchain.agents import tool

class ConfigToolset():
    @tool("LoadHistoricalTrainingLogs")
    def load_historical_training_logs(file_path: str):
        "This tool is designed for easily loading and reviewing model training logs. It automatically accesses records of loss and accuracy metrics from different hyperparameter settings."
        if os.path.exists(file_path):
            with open(file_path) as f:
                logs = json.load(f)
            return logs

    @tool("LoadConfigs")
    def load_configs(file_path: str):
        "Useful for when you need to loading the model training configs and read the content. The file contains the hyperparameters that used to define the training details of the model."
        if os.path.exists(file_path):
            with open(file_path) as f:
                configs = json.load(f)
            return configs

    @tool("WriteConfigs")
    def write_configs(file_path: str, configs: dict):
        "Useful for when you need to writing the changed configs into file. Input should be the hyperparameters that you want to write into the file IN JSON FORMAT. And you should also keep the unchanged Hyperparameter into the file."
        with open(file_path, 'w') as f:
            json.dump(configs, f)
        return f"Successfully wrote configs to {file_path}"

    @tool("ExecutePythonFile")
    def execute_python_file(file_path: str):
        "Useful for when you need to execute the python file to training the model"
        os.system(f"python {file_path}")
        return f"Successfully executed {file_path}"

    @tool("LoadTrainingLogs")
    def load_training_logs(file_path: str):
        "Useful for when you need to loading the model training logs and read the content. The file contains the training logs (loss, accuracy) generated by training."
        if os.path.exists(file_path):
            with open(file_path) as f:
                logs = f.read()
            return logs