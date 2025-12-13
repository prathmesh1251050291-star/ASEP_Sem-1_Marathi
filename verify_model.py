import pickle
import os

try:
    if not os.path.exists('ASEP_Crop.pkl'):
        print("Error: ASEP_Crop.pkl not found")
        exit(1)
        
    with open('ASEP_Crop.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Success: Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
