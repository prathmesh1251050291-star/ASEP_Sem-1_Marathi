import os

file_path = 'ASEP_Crop.pkl'
try:
    with open(file_path, 'rb') as f:
        header = f.read(50)
    print(f"Header bytes: {header}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
except Exception as e:
    print(f"Error reading file: {e}")
