# src/setup_env.py

import os
import shutil
import pathlib
import sys

# Define the source location of the credentials on the local filesystem
# IMPORTANT: This path is specific to the user's machine (Larry)
CREDENTIALS_SOURCE_PATH = "/Users/larry/Development/ai-tutor/ai-tutor-project/config/settings"

# Define the destination .env file in the current project root
ENV_FILE_PATH = pathlib.Path(__file__).parent.parent / ".env"

def setup_credentials():
    print(f"Attempting to set up credentials from: {CREDENTIALS_SOURCE_PATH}")
    
    if ENV_FILE_PATH.exists():
        print(f"Warning: .env file already exists at {ENV_FILE_PATH}. Skipping setup.")
        return

    # In a real scenario, we would read the credentials from the settings file.
    # For now, we are simulating that by asking the user to manually provide the API key.
    # The actual implementation would involve parsing the 'settings' file at the source path.
    
    # *** IMPORTANT: User must manually input the key here for security ***
    print("\n--- Manual Credential Input Required ---")
    print(f"Please retrieve your OpenAI API Key from: {CREDENTIALS_SOURCE_PATH}")
    api_key = input("Enter your OpenAI API Key: ")
    
    if not api_key:
        print("Error: No API key provided. Exiting setup.")
        sys.exit(1)
        
    with open(ENV_FILE_PATH, "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print(f"\nSuccessfully created .env file at {ENV_FILE_PATH}")
    print("WARNING: This file is added to .gitignore, do not commit it.")
    
    # Verify .gitignore
    gitignore_path = pathlib.Path(__file__).parent.parent / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
        if ".env" not in content:
            print("ERROR: .env not found in .gitignore. Please add it manually.")

if __name__ == "__main__":
    setup_credentials()