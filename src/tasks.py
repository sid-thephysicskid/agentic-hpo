from textwrap import dedent
from crewai import Task
from tools import *
import json

class HPTuningTasks:
    def __init__(self, config_file: str, experiment_logs_file: str):
        self.config_file = config_file
        self.experiment_logs_file = experiment_logs_file
        self.historical_results_tool = HistoricalResultsTool(config_file=self.config_file)
        self.python_execution_tool = PythonExecutionTool(config_file=self.config_file)
        # self.write_configs_tool = WriteConfigs(config_file=self.config_file)

    def load_config(self):
        try:
            with open(self.config_file) as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading configuration file: {e}")
            return None
        
    def load_experiment(self):
        try:
            with open(self.experiment_logs_file) as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading experiment logs file: {e}")
            return None
    
    def creator_task(self, agent):
        if self.config_file is None:
            print("Configuration file is not loaded properly.")
            return
        if self.experiment_logs_file is None:
            print("Experiment logs file is not loaded properly.")
            return
        config_data = self.load_config()
        experiment_data = self.load_experiment()
        if 'experiments' not in experiment_data:
            experiment_data['experiments'] = []

        task_description = dedent(f"""
        Check the previous hyperparameter tuning plan and completed tasks results from the 'experiments' key in  {self.experiment_logs_file} using {self.historical_results_tool} tool.
        
        If no previous experiments are found, create the first experiment with your best guess for hyperparameters based on the provided information.
        
        Below is the basic information about the experimental settings:
        Model Info: {config_data["model_info"]}
        Dataset Info: {config_data["dataset_info"]}
        Hyperparameter search space: {config_data["hyperparameter_info"]}
        Objective: {config_data["optim_goal"]}

        Based on this information, generate a new sub-task for the task execution agent that can solve the sub-task.
        
        Format your response as follows:
        Objective: Define the final goal
        Thought: Describe your reasoning process
        Action: Specify the action to take, valid actions are `Final Answer` or {self.historical_results_tool}
        Action Input: Input for the action
        Observation: Outcome of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: The proposed hyper-parameters for the task
        
        Propose a new task focused on unexplored hyperparameter spaces or optimization techniques to methodically reach the final objective. 
        The task executor will adjust hyperparameters and run the training script.
        Ensure your proposed hyperparameters are distinct from those previously tested, and state your recommendation as the `Final Answer`.
        When providing your final answer, structure the suggested hyperparameters and reasoning as follows:
        Please provide your suggestions in the following dictionary structure:
        {{
            'suggested_hyperparameters': {{
                "learning_rate": <value>,
                "max_depth": <value>,
                "n_estimators": <value>,
                "subsample": <value>
            }},
            'creator_reasoning': "Explain your choices here, mentioning why each hyperparameter was chosen based on the previous results and the hyperparameter search space."
        }}
        
        """)
        # Use {self.write_configs_tool}  tool to log new hyperparameters and reasoning to {self.config_file} under 'experiments'.
        expected_output = f"A dictionary with 'suggested_hyperparameters' and 'creator_reasoning' as keys, and their values matching the format provided in the task description."
        return Task(
            agent=agent,
            description=task_description,
            expected_output=expected_output,
            tools = [self.historical_results_tool],
            # callback= log_final_answer,
            # output_file=self.experiment_logs_file
        )

    def executor_task(self, agent):
        if self.config_file is None:
            print("Configuration file is not loaded properly.")
            return
        print(f"config file {self.config_file}")
        config_data = self.load_config()
        print(f"tuning script is motherfucking supposed to be {config_data['tuning_script']}")
        experiment_data = self.load_experiment()
        task_description = dedent(f"""
        Read the latest 'proposed_hyperparameters' and 'creator_reasoning' from 'experiments' in {experiment_data}.
        Execute the model tuning script `{config_data["tuning_script"]}` with the proposed hyperparameters using {self.python_execution_tool} tool.
        
        Use the following format:
        Task: the input task you must solve
        Thought: you should always think about what to do
        Action: the action to take, should be {self.python_execution_tool}
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        After finishing the task, analyze the training logs to make a summary about this experiment, including the analysis of the
        training trajectory and final training results. Then provide your final answer.
        Please provide your suggestions in the following dictionary structure:
        {{
            'hyperparameters_tried': "The hyperparameters passed to the tuning script",
            'executor_reasoning': "Briefly explain the results of running the tuning script with the proposed hyperparameters and any insights gained",
        }}
        """)
        expected_output = f"A dictionary with 'hyperparameters_tried' and 'executor_reasoning' as keys, and their values matching the format provided in the task description."
        return Task(
            agent=agent,
            description=task_description,
            expected_output=expected_output,
            tools = [self.python_execution_tool],
            # callback=log_executor_results,
            # output_file=self.experiment_logs_file
        )
