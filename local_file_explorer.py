import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# List all files in the current directory and subdirectories
import os
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can also list specific directories like 'model' folder
print("\nFiles in model directory:")
if os.path.exists('model'):
    for filename in os.listdir('model'):
        print(os.path.join('model', filename))
else:
    print("Model directory does not exist") 