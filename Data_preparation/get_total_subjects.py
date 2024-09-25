import h5py
import sys
import pandas as pd
def get_total_subjects(file_path):
    with h5py.File(file_path, 'r') as f:
        # Adjust this based on the actual structure of your file
        # Here 'total_subjects' is assumed to be a dataset or attribute
        # if 'total_subjects' in f:
        #     total_subjects = f['total_subjects'][()]
        # elif 'some_dataset' in f:  # Replace with actual dataset name
        #     total_subjects = f['some_dataset'].shape[0]
        # else:
        #     raise ValueError("Unable to find total_subjects or equivalent dataset.")
        df= pd.read_hdf(file_path)
        total_subjects =len(df)+1
    return total_subjects

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_total_subjects.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    total_subjects = get_total_subjects(dataset_path)
    print(total_subjects)
    # sys.exit(0)

