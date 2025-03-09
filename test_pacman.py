import subprocess

# List of agents to test
agents = [
    "ReflexAgent",
    "MinimaxAgent",
    "AlphaBetaAgent",
    "RandomExpectimaxAgent",
    "DirectionalExpectimaxAgent",
    "CompetitionAgent"
]

# List of layouts to test
layouts = [
    "testClassic",
    "smallClassic",
    "mediumClassic"
]

# Number of games to run (reduced for quicker testing)
num_games = 3

# Number of ghosts to test with
ghost_counts = [1, 2, 4]

# Function to run a command and capture the output
def run_command(command, agent):
    print(f"Running: {command}")
    print(f"Agent used: {agent}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

# Testing all combinations of agents, layouts, and ghost counts
for agent in agents:
    for layout in layouts:
        for ghost_count in ghost_counts:
            command = f"python pacman.py -p {agent} -l {layout} -k {ghost_count} -n {num_games} -q"
            run_command(command, agent)

print("Testing complete.")
