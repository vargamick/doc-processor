#!/bin/bash

# Install required Python packages if not already installed
pip install pyyaml >/dev/null 2>&1 || { echo "Error installing dependencies"; exit 1; }

# Make the Python script executable
chmod +x validate_docker_setup.py

# Run the validation script
python3 validate_docker_setup.py

# Exit with the same status as the Python script
exit $?
