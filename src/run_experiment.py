import argparse
import os
import sys
from pathlib import Path
from importlib import import_module

parser = argparse.ArgumentParser(prog="run-experiment", exit_on_error=False)

parser.add_argument("experiment_name", help="Either the full name of the experiment folder or, if using -n, just the number of the folder.")
parser.add_argument(
    "-n", "--number-id", action="store_true"
) 

try:
    args = parser.parse_args()
except argparse.ArgumentError:
    print("Warning : No argument passed for experiment_name. Trying to run experiment E01_...")
    args = argparse.Namespace(experiment_name = "01", number_id = True)

experiment_name = args.experiment_name

if args.number_id :
    experiment_name = int(experiment_name)
    exp_folders = []
    for folder in os.listdir("./experiments"):
        print("Folder : ", folder, " , matching pattern : ", f"E{experiment_name:02d}_")
        if folder.startswith(f"E{experiment_name:02d}_"):
            exp_folders.append(folder)
    if len(exp_folders) != 1:
        raise FileNotFoundError(f"You have {len(exp_folders)} matching \"E{experiment_name}_\". Please renumber appropriatly your experiment folders.")
    experiment_name = exp_folders[0]




# Get the parent directory
global_model_package_path = Path(__file__).resolve().parent.joinpath("global_model_package")
# Add global_model_package directory to sys.path
sys.path.append(str(global_model_package_path))
print(sys.path)

print("Running 'run_single_simu.py', here we goooo !")
import_module(f"experiments.{experiment_name}.run_single_simu")  #run the run_single_simu file


print("Experiment finished.")

