import os
import re

def delete_checkpoints_except_n_highest_steps(folder_path="model_checkpoints", n=1):
    """
    Deletes all files in the given folder except for the n files with the highest steps.

    Args:
        folder_path (str): Path to the folder containing the files.
        n (int): Number of files to keep.
    """
    # Get a list of all files in the folder
    files = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename)) and filename.startswith("model_pg_") and filename.endswith("_steps.pth")]
    interrupted_files = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename)) and filename.startswith("model_pg_") and filename.endswith("_interrupted.pth")]
    
    # print (f'all files: {files}')
    

    # Extract the step number from each file name and store it in a dictionary
    file_steps = {}
    for file in files:
        match = re.search(r'model_pg_(\d+)_steps\.pth', file)
        if match:
            steps = int(match.group(1))
            file_steps[file] = steps

    # Sort the files by their step numbers in descending order
    sorted_files = sorted(file_steps, key=file_steps.get, reverse=True) # short in descending order
    files_to_delete = sorted_files[n:]
    # print(f'files to delete: {files_to_delete}')
    # return
    
    # Delete all files except for the n files with the highest steps
    for file in files_to_delete:
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted file: {file}")
    
    # Delete all interrupted files
    for file in interrupted_files:
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted file: {file}")



# def get_max_epoch_file(directory='model_checkpoints'):
#     max_epoch = 0
#     max_epoch_file = None
    
#     if os.path.exists(directory):
#         for filename in os.listdir(directory):
#             if filename.startswith("model_pg_epoch_") and filename.endswith(".pth"):
#                 try:
#                     epoch = int(filename.split("_")[-1].split(".")[0])
#                     if epoch > max_epoch:
#                         max_epoch = epoch
#                         max_epoch_file = filename
#                 except Exception as Ex:
#                     print(f'file: {filename} : {Ex}')
    
#     return os.path.join(directory, max_epoch_file) if max_epoch_file else None

def get_max_global_step_file(directory='model_checkpoints'):
    max_step = 0
    max_steps_file = None

    if os.path.exists(directory):
        for filename in os.listdir(directory):
            # format: model_pg_{global_step}_steps.pth
            if filename.startswith("model_pg_") and filename.endswith("_steps.pth"):
                try:
                    step = int(filename.split("model_pg_")[-1].split("_steps.pth")[0])
                    if step > max_step:
                        max_step = step
                        max_steps_file = filename
                except Exception as Ex:
                    print(f'file: {filename} : {Ex}')
    
    return os.path.join(directory, max_steps_file) if max_steps_file else None