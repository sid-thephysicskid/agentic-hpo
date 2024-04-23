import os
import json
import subprocess
from crewai_tools import BaseTool

class HistoricalResultsTool(BaseTool):
    name: str = "LoadHistoricalResults"
    description: str = "Loads and reviews model training results from a configuration file."

    def _run(self, experiment_logs_file: str) -> str:
        if os.path.exists(experiment_logs_file):
            with open(experiment_logs_file) as f:
                logs = json.load(f)
            return logs.get("experiments", "No historical experiments found.")
        else:
            return f"File not found: {experiment_logs_file}"

class PythonExecutionTool(BaseTool):
    name: str = "ExecutePythonFile"
    description: str = "Executes a Python script as specified in a configuration file with the latest parameters."

    def _run(self, config_file:str, experiment_logs_file: str) -> str:
        

        if os.path.exists(experiment_logs_file):
            with open(experiment_logs_file) as f:
                experiment_data = json.load(f)
                print(f"Experiment data: {experiment_data}")
        latest_experiment = experiment_data["experiments"][-1]
        hyperparameters = latest_experiment.get("suggested_hyperparameters",{})
        if not hyperparameters:
            return "No hyperparamters found in the latest experiment"
        
        command = [
            "python",
            config_data["tuning_script"],
            config_data["data_file"],
            config_data["target_column"],
        ]

        print(f"Command is {command}")
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            return f"Successfully executed {config_data['tuning_script']}: {result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Failed to execute {config_data['tuning_script']}: {e}.\nOutput: {e.stdout}\nError: {e.stderr}"

def read_json(file_path):
    """Reads a JSON file and returns the data."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json(file_path, data):
    """Writes data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def log_final_answer(step_details):
    """Logs only the final answer details to 'experiment_logs.json' under 'experiments'."""
    file_path = 'experiment_logs.json'
    config_data = read_json(file_path)
    if config_data is None:
        config_data = {'experiments': []}  # Initialize if file does not exist

    experiment_id_counter = get_last_experiment_id(config_data) + 1
    for detail in step_details:
        if isinstance(detail, tuple) and len(detail) > 1:
            data_part = detail[1]
            if isinstance(data_part, dict) and 'output' in data_part:
                try:
                    output = json.loads(data_part['output'].replace("'", "\""))
                    if 'suggested_hyperparameters' in output and 'creator_reasoning' in output:
                        final_answer = {
                            'experiment_id': experiment_id_counter,
                            'suggested_hyperparameters': output['suggested_hyperparameters'],
                            'creator_reasoning': output['creator_reasoning']
                        }
                        config_data['experiments'].append(final_answer)
                        write_json(file_path, config_data)
                        experiment_id_counter += 1  # Increment for the next entry
                except json.JSONDecodeError as e:
                    print("JSON decoding error:", e)


def log_executor_results(step_details):
    """Logs executor details to the same experiment entry in 'experiment_logs.json'."""
    file_path = 'experiment_logs.json'
    config_data = read_json(file_path)
    if config_data is None:
        print("Experiment log file not found or is empty.")
        return

    experiment_id = get_last_experiment_id(config_data)
    for detail in step_details:
        if isinstance(detail, tuple) and len(detail) > 1:
            data_part = detail[1]
            if isinstance(data_part, dict) and 'executor_output' in data_part:
                try:
                    output = json.loads(data_part['executor_output'].replace("'", "\""))
                    if 'hyperparameters_tried' in output and 'executor_reasoning' in output:
                        executor_data = {
                            'experiment_id': experiment_id,
                            'hyperparameters_tried': output['hyperparameters_tried'],
                            'executor_reasoning': output['executor_reasoning']
                        }
                        # Find the matching experiment by ID and update it
                        for experiment in config_data['experiments']:
                            if experiment.get('experiment_id') == experiment_id:
                                experiment.update(executor_data)
                                write_json(file_path, config_data)
                                break
                except json.JSONDecodeError as e:
                    print("JSON decoding error:", e)

def get_last_experiment_id(config_data):
    """Extracts the last experiment ID from the config data."""
    experiments = config_data.get("experiments", [])
    if not experiments:
        return 0
    return experiments[-1].get('experiment_id', 0)