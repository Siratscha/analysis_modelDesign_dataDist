import os
import csv
def get_small_files(directory_paths, max_size_kb=5):
    small_files = []
    
    for directory_path in directory_paths:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size_kb = os.path.getsize(file_path) / 1024  # Convert file size to KB
                
                if file_size_kb < max_size_kb:
                    small_files.append(file_path)
    
    return small_files

# Example usage:
directory_paths = ["/work/srankl/thesis/modelDesign_bias_CXR/data/MIMICCXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files/22/22026000/22026000/", 
"/work/srankl/thesis/modelDesign_bias_CXR/data/MIMICCXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files/22/22045000/22045000/", 
"/work/srankl/thesis/modelDesign_bias_CXR/data/MIMICCXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files/20/20030000/20030000/"]
small_files_list = get_small_files(directory_paths) 


def save_to_csv(file_list, csv_file):
    with open(csv_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Path"])
        for file_path in file_list:
            writer.writerow([file_path])

cwd = r'/work/srankl/thesis/development/modelDesign_bias_CXR/' 
csv_file = "small_files"
path = os.path.join(cwd,csv_file)  + ".csv"
save_to_csv(small_files_list, path)
