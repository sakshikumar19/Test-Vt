import csv
import json
import argparse

def is_number(value):
    """Check if a value is a number (integer or float)."""
    try:
        float(value)  # Try converting to float
        return True
    except ValueError:
        return False

def csv_to_json(csv_file, json_file):
    data = []
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in ['expected_chunks', 'retrieved_chunks', 'accuracy', 'found_chunks', 'total_expected']:
                if key in row and is_number(row[key]):  
                    row[key] = int(row[key]) if row[key].isdigit() else float(row[key])
            data.append(row)
    
    with open(json_file, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    
    print(f"CSV converted to JSON and saved as {json_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to JSON")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("json_file", help="Path to save the JSON file")
    args = parser.parse_args()
    
    csv_to_json(args.csv_file, args.json_file)
