import numpy as np
import re
import csv
import os


def load_csv(filename, sep=';', skiprows=0):
    return np.loadtxt(open(filename, "rb"), delimiter=sep, skiprows=skiprows)

def load_cross_section(filename):
    """Takes a text file with two columns of numbers and return two arrays : one for each column"""
    data_cs = load_csv(filename)
    return data_cs[:,0], data_cs[:,1]




def parse_cross_sections(file_path, reaction_name, output_dir, reaction_name_in_file=None):
    if reaction_name_in_file is None:
        reaction_name_in_file = reaction_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    block_index = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect reaction block start
        if reaction_name in line:
            reaction_type = line
            energy_line = lines[i + 2].strip()
            try:
                energy_value = float(energy_line)
            except ValueError:
                raise ValueError("The energy threshold can't be converted to a float")

            # Find start of data table
            while i < len(lines) and not re.match(r"^\s*-{5,}", lines[i]):
                i += 1
            i += 1  # skip dashed line

            # Read the data table
            data = []
            while i < len(lines) and not re.match(r"^\s*-{5,}", lines[i]):
                parts = lines[i].strip().split()
                if len(parts) == 2:
                    data.append(parts)
                i += 1

            # Prepare filename
            clean_name = re.sub(r"[^\w\-_\. ]", "_", reaction_name.replace("->", "to"))
            filename = f"{reaction_type}_{clean_name}_E={energy_value:.4f}eV.csv"
            filepath = os.path.join(output_dir, filename)

            # Write CSV
            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Energy (eV)", "Cross section (m²)"])
                for row in data:
                    writer.writerow(row)

            block_index += 1

        i += 1  # move to next line

    print(f"Done. Extracted {block_index} reactions to '{output_dir}'.")

# Example usage
#parse_cross_sections("crosssections.txt", "output_csvs")



if __name__ == "__main__":
    e_r,cs_r=load_cross_section('..\\cross_sections\\Xe\\exc_Xe.csv')
    print(e_r)
    print(cs_r)
    print("done")