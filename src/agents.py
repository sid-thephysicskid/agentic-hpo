
# from functools import partial
from textwrap import dedent
from crewai import Agent
from tools import * 

class HPTuningAgents:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.historical_results_tool = HistoricalResultsTool(config_file=self.config_file)
        self.python_execution_tool = PythonExecutionTool(config_file=self.config_file)

    def load_config(self):
        try:
            with open(self.config_file) as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading configuration file: {e}")
            return None

    def creator_agent(self):
        tools = [self.historical_results_tool]
        # print("Debug: Tools List -", tools)
        try:
            creator= Agent(
                role="Hyperparameter configuration creator",
                backstory=dedent("""\
                                You are a task creation AI expert in machine learning that is required
                                to optimize the hyperparameter settings of accomplish the final objective."""
                                ),
                goal="Interpret task-specific background information and generate new hyperparameter configuration",
                tools=tools,
                step_callback=log_final_answer,
                verbose=True,
                allow_delegation=True
            )
            # print("Debug: Creator Agent Tools -", creator.tools)
            return creator
        except Exception as e:
            print("Error creating creator agent: ", e)
            return None
    def executor_agent(self):

        tools = [self.python_execution_tool]
        executor= Agent(
            role="Model tuning executor",
            backstory=dedent("""\
                            You are the machine learning experimenter whose task is to finish the given objective below. """),
            goal="Employ the configuration sent by the creator to train models, analyze the training outputs, and log the experiment data",
            tools=tools,
            step_callback= log_executor_results,
            verbose=True,
            allow_delegation=True
        )
        # print("Debug: Executor Agent Tools -", executor.tools)
        return executor