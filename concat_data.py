import os
import pandas as pd
import sys

def concat_data_files(directory):
    # Get the list of Excel files in the directory
    data_files = [f for f in os.listdir(directory) if f.endswith('.ftr')]

    # Read and concatenate the files
    dict_df = {}
    for idx, file in enumerate(data_files):
        print("Loading {} file out of {} files".format(idx+1, len(data_files)))
        file_path = os.path.join(directory, file)
        df = pd.read_feather(file_path)
        dict_df[file] = df

    # Concatenate the DataFrames
    concatenated_df = pd.concat(dict_df.values())

    # Save the concatenated DataFrame to a new Excel file
    output_file = os.path.join(directory, 'DeepIV v2.0.0.ftr')
    concatenated_df.reset_index(drop=True).to_feather(output_file)

    print("Concatenated file has shape of:", concatenated_df.shape)
    print(f"Concatenated file saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python concat_excel.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    concat_data_files(directory)
