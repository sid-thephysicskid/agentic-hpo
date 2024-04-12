from textwrap import dedent
from crewai import Agent

from agentic_hyperparameter_optimization.src.tools import*

class HPTuningAgents():
    def creator_agent(self):
        return Agent(
            role="Hyperparameter configuration creator",
            backstory=dedent("""\
                            You are a task creation AI expert in machine learning that is required
                            to optimize the hyperparameter settings of the model"""),
            # goal="Optimize the hyperparameter settings of the model to accomplish the final objective"
            goal= "Interpret task-specific background information and generate new hyperparameter configuration",
            tools=[],
            verbose=True
        )
    def executor_agent(self):
        return Agent(
            role="Model tuning executor",
            backstory=dedent("""\
                             You are the machine learning experimenter whos is asked to finish the given objective below. """),
            goal="Employ the configuration sent by Hyperparameter configuration creator to train models, analyze the training outputs, and log the experimental data",
            tools=[],
            verbose=True
        )
    

#You are a {task.agent.role}. {task.agent.backstory}. Your goal is {task.agent.goal}. 
#You have the following tools at your disposal: {task.agent.tools}. 
#Your task is to {task.description}.