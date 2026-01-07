import logging
import sys
from src import data_loader, visualization

# Configure Logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("test_script")

try:
    logger.info(">>> STEP 1: Loading Data...")
    raw_train, raw_test = data_loader.load_and_process_subject(1)
    
    logger.info(">>> STEP 2: Generating Visualizations for Presentation...")
    
    # Plot 1: Did the filter work? (Should see a drop off after 38Hz)
    visualization.plot_power_spectrum(raw_train)
    
    # Plot 2: What does the signal look like?
    visualization.plot_raw_segment(raw_train)
    
    print("\n>>> SUCCESS! Check the 'results' folder for your images!")

except Exception as e:
    logger.exception("Something went wrong!")