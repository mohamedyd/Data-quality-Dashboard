import subprocess


# Function to check if PostgreSQL is running (required for HoloClean)
def is_postgresql_running():
    try:
        # Run the command to check the PostgreSQL service status
        result = subprocess.run(['service', 'postgresql', 'status'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # Check if the service is active (running)
        return "online" in result.stdout
    except subprocess.CalledProcessError:
        # If the command failed, PostgreSQL is not running
        return False

commands = [
    "python3 -m main.app",  # Command for running the dashboard
    "uvicorn fastAPImain:app --reload"  # Command for running the API
]

# Check if PostgreSQL is running
if is_postgresql_running():
    processes = []

    for command in commands:
        processes.append(subprocess.Popen(command, shell=True, cwd="API" if "uvicorn" in command else None))

    for process in processes:
        process.wait()

else:
    print("PostgreSQL service is not running. Please start the service and try again.")

