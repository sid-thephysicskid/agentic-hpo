

import json
import os
from dotenv import load_dotenv
from crewai import Crew
from agents import HPTuningAgents
from tasks import HPTuningTasks

def main():
    load_dotenv()  # Load environment variables if needed

    config_file = "config.json"
    experiment_logs_file= "experiment_logs.json"
    
    agents = HPTuningAgents(config_file)
    creator_agent = agents.creator_agent()
    executor_agent = agents.executor_agent()
    # Instantiate tasks
    tasks = HPTuningTasks(config_file, experiment_logs_file)
    # # Create tasks
    create_task = tasks.creator_task(creator_agent)
    execute_task = tasks.executor_task(executor_agent)
    
    crew = Crew(
        agents=[creator_agent, executor_agent],
        tasks=[create_task, execute_task]
    )

    result = crew.kickoff()
    
    print("Crew execution result:", result)

def setup_config_file(config_file):
    """Ensure the configuration file exists and is properly initialized."""
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump({"experiments": []}, f, indent=4)
    else:
        with open(config_file, 'r+') as f:
            data = json.load(f)
            if 'experiments' not in data:
                data['experiments'] = []
                f.seek(0) #resets the file's cursor to the beginning
                json.dump(data, f, indent=4) #indent is to make it more readable
                f.truncate() #used to cut off the file at the current position of the cursor (which is at the end of the newly written JSON data)
    return data

def load_config(self):
    """Load configuration data from a JSON file."""
    try:
        with open(self.config_file, 'r') as file:
            return json.load(file)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading configuration file: {e}")
        return None  # or handle the error as needed


def get_next_experiment_id(config_file):
    """Get the next experiment ID based on the existing entries in the config file."""
    with open(config_file, 'r') as f:
        data = json.load(f)
        experiments = data.get('experiments', [])
        if experiments:
            return max(exp.get('experiment_id', 0) for exp in experiments) + 1
        return 1  # Start from 1 if no experiments exist

if __name__ == "__main__":
    main()
