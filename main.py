# # from app.flowsync_app import run


# import os
# import sys

# # Outer project directory
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# # Inner FlowSync directory containing ML_Module and controller
# FLOWSYNC_DIR = os.path.join(PROJECT_ROOT, "FlowSync")

# # Make both folders available for imports
# sys.path.insert(0, PROJECT_ROOT)
# sys.path.insert(0, FLOWSYNC_DIR)

# from ML_Module.predict_live import run


# if __name__ == "__main__":
#     run()

import os
import sys

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure root folder is available for imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ML_Module.predict_live import run


if __name__ == "__main__":
    run()