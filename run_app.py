
import subprocess
import sys
import os

def main():
    # Determine the path to main.py
    ## if _internal folder exists put _internal then app
    if os.path.exists('_internal'):
        script_path = os.path.join('_internal','app','main.py')
    else:
        script_path = os.path.join('app','main.py')

    # Build the command to run the Streamlit app
    print(f'Running Streamlit app: {script_path}')

    cmd = [sys.executable, '-m', 'streamlit', 'run', script_path]

    # Run the Streamlit app
    subprocess.run(cmd)

if __name__ == '__main__':
    main()