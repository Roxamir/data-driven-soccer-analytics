# Citations:
# [1] https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
# [2] https://stackoverflow.com/questions/64512076/python-load-json-lines-to-pandas-dataframe

import json
import pandas as pd

def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flatten a nested dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # If the value is a list of dictionaries, flatten each dictionary
            if v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
            else:
                # For lists of non-dictionaries, convert to string
                items.append((new_key, str(v)))
        else:
            # For simple values, keep as is
            items.append((new_key, v))
    
    return dict(items)

def load_json_to_dataframe(filename):
    # Load the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # If data is a dictionary with a list, try to extract that list
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                data = value
                break
    
    # Flatten each record in the data
    flattened_data = [flatten_dict(record) for record in data]
    
    # Convert to DataFrame
    dataframe = pd.DataFrame(flattened_data)
    
    return dataframe

# Input JSON file
# WHEN CONVERTING A DIFFERENT FILE, ENSURE THAT INPUT FILENAME MATCHES OUTPUT FILENAME
datafile = 'players_data_39_2021_min10.json'
dataframe = load_json_to_dataframe(datafile)

# Print column names to verify
print("Columns:", list(dataframe.columns))

# Save the dataframe to a CSV file (CHECK TO MAKE SURE THE FILENAME IS CORRECT)
output_file = 'players_data_39_2024_min10.csv'
dataframe.to_csv(output_file, index=False)
print(f"CSV file saved as {output_file}")

# Print first few rows to verify
print("\nFirst few rows:")
print(dataframe.head())