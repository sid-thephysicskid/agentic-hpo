from dotenv import load_dotenv
from crewai import Crew
from tasks import HPTuningTasks
from agents import HPTuningAgents


def main():
    load_dotenv()
    tasks = HPTuningTasks()
    agents = HPTuningAgents()

    #create agents
    creator_agent = agents.creator_agent()
    executor_agent = agents.executor_agent()

    #create tasks
    create = tasks.creator_task(creator_agent, model_info, dataset_info, hyperparameter_info, tool_names, optim_goal, agent_scratchpad)
    execute = tasks.executor_task(executor_agent, tool_names, task_name, agent_scratchpad)

    crew = Crew(
        agents=[
            creator_agent,
            executor_agent
        ],
        tasks=[
            create,
            execute
        ],
        process='sequential'
    )

    result=crew.kickoff()

if __name__=="__main__":
    main()