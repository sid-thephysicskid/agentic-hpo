from textwrap import dedent
from crewai import Task

class HPTuningTasks():
    def creator_task(self, agent, model_info, dataset_info, hyperparameter_info, tool_names, optim_goal, agent_scratchpad):
        return Task(
            description=dedent(f"""\
                                Check the previous hyperparameter tuning plan and completed tasks results.
                                Based on this information, generate a new sub-task for the task execution agent that can solve the sub-task. 
                                Below is the basic information about the experimental settings:
                                Model info: {model_info}
                                Dataset info: {dataset_info}
                                Below is the hyper-parameters and corresponding candidates or values range that can be tuned for the task:
                                {hyperparameter_info}
                                Format your response as follows:
                                Objective: Define the final goal
                                Thought: Describe your reasoning process
                                Action: Specify the action to take; valid actions are `Final Answer` or {tool_names}
                                Action Input: Input for the action
                                Observation: Outcome of the action
                                ... (this Thought/Action/Action Input/Observation can repeat N times)
                                Thought: I now know the final answer
                                Final Answer: The proposed hyper-parameters for the task
                                
                                Analyze the completed tasks and their outcomes.
                                Propose a new task focused on unexplored hyperparameter spaces or
                                optimization techniques to methodically reach the final objective.
                                The task executor will adjust hyperparameters and run the training script.
                                Ensure your proposed hyperparameters are distinct from those previously tested,
                                and state your recommendation as the `Final Answer`.
                                Objective: {optim_goal}
                                Thought: {agent_scratchpad}"""),
            expected_output=dedent(f"""\
                                   A new task focused on unexplored hyperparameter spaces or
                                   optimization techniques to methodically reach the final objective."""),
            async_execution = False,
            agent=agent
        )
    
    def executor_task(self, agent, tool_names, task_name, agent_scratchpad):
        return Task(
            description=dedent(f"""\
                                Finish the given objective below. Use the following format:
                                Task: the input task you must solve
                                Thought: you should always think about what to do
                                Action: the action to take, should be one of [{tool_names}]
                                Action Input: the input to the action
                                Observation: the result of the action
                                ... (this Thought/Action/Action Input/Observation can repeat N times)
                                Thought: I now know the final answer
                                Final Answer: the final answer to the original input question
                                After finishing the task, analyze the training logs to make a summary about this experiment,
                                including the analysis of the training trajectory and final training results.
                                Then provide your answer with Final Answer.
                                Task: {task_name}
                                Thought:{agent_scratchpad}
                               """),
            expected_output=dedent(f"""\
                                A summary about this experiment, including the analysis of the
                                training trajectory and final training results."""),
            async_execution = False,
            agent=agent
        )