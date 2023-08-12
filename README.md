# helpers
Common helper files for reusability across PyTorch, Spark, Sklearn, Pandas, and more

## Example of downloading a helper file

import requests
from pathlib import Path 

HELPER_NAME = 'plot_decision_boundary'
HELPER_PATH = 'https://raw.githubusercontent.com/gputrain/helpers/main/PyTorch/plot_decision_boundary.py'

if Path(HELPER_NAME+'.py').is_file():
    print(HELPER_NAME+'.py'  + " already exists, skipping download")
else:
    print("Downloading "+ HELPER_NAME+'.py')
    request = requests.get(HELPER_PATH)
    with open(HELPER_NAME+'.py', "wb") as f:
        f.write(request.content)
        
        
### Using the import in a function

from plot_decision_boundary import plot_decision_boundary