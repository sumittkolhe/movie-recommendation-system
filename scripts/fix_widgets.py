import sys
import subprocess
import os

def run_command(command):
    """Run a command and return True if successful"""
    print(f"Running: {command}")
    try:
        # Use list form for subprocess to handle spaces in paths properly
        if isinstance(command, str):
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def fix_ipywidgets():
    """Fix common ipywidgets installation issues"""
    print("Starting ipywidgets installation fix...")
    
    python_exe = sys.executable
    # Wrap in quotes for Windows paths with spaces
    python_exe_quoted = f'"{python_exe}"'
    
    # Step 1: Install/upgrade required packages
    packages = [
        "notebook",
        "ipywidgets",
        "jupyterlab",
        "widgetsnbextension"
    ]
    
    for package in packages:
        print(f"\n=== Installing/upgrading {package} ===")
        run_command(f"{python_exe_quoted} -m pip install --user --upgrade {package}")
    
    # Step 2: Enable notebook extension
    print("\n=== Enabling notebook extension ===")
    run_command(f"{python_exe_quoted} -m jupyter nbextension enable --py widgetsnbextension")
    
    # Step 3: For JupyterLab users (optional)
    print("\n=== Installing JupyterLab extension ===")
    run_command(f"{python_exe_quoted} -m pip install --user jupyterlab_widgets")
    
    # Step 4: Fix possible path issues
    print("\n=== Adding Scripts directory to PATH ===")
    user_scripts_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python313", "Scripts")
    
    if os.path.exists(user_scripts_path):
        print(f"Adding {user_scripts_path} to PATH temporarily")
        if user_scripts_path not in os.environ["PATH"]:
            os.environ["PATH"] = user_scripts_path + os.pathsep + os.environ["PATH"]
    
    # Final verification
    print("\n=== Verifying Installation ===")
    run_command(f'{python_exe_quoted} -c "import ipywidgets; print(\'ipywidgets version:\', ipywidgets.__version__)"')
    
    # Instructions for user
    print("\n=== INSTRUCTIONS ===")
    print("1. Close and reopen Jupyter Notebook/Lab")
    print("2. Try running your notebook again")
    print("3. If still not working, try running Jupyter with:")
    print(f"   {python_exe_quoted} -m jupyter notebook")
    print("\nIf issues persist, you can use the standalone movie_recommendation.py script instead.")

if __name__ == "__main__":
    fix_ipywidgets() 