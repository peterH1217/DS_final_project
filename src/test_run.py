import logging
from src.data_loader import load_and_process_subject
from src.visualization import plot_psd, plot_raw_trace

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info(">>> STEP 1: Starting Multi-Dataset Pipeline...")
    
    # 1. Defining the list of datasets we want to process
    # 'BNCI2014_001' = BCI Competition IV 2a
    # 'Schirrmeister2017' = High Gamma Dataset (HGD)
    datasets_to_run = ['BNCI2014_001', 'Schirrmeister2017']
    
    for dataset_name in datasets_to_run:
        logger.info(f"\n--- PROCESSING DATASET: {dataset_name} ---")
        
        try:
            # 2. Loading Data (Dynamic Switch)
            raw_train, raw_test = load_and_process_subject(1, dataset_name=dataset_name)

            # 3. Generating Visualizations (with unique filenames!)
            logger.info(f"Generating visualizations for {dataset_name}...")
            
            # We add the dataset_name to the filename so they don't overwrite each other
            plot_psd(
                raw_train, 
                save_path=f"results/psd_plot_{dataset_name}.png"
            )
            
            plot_raw_trace(
                raw_train, 
                save_path=f"results/raw_eeg_trace_{dataset_name}.png"
            )
            
            logger.info(f"Successfully finished {dataset_name}. Check results folder!")

        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")

    logger.info("\n>>> Pipeline complete: we have results for both datasets.")

if __name__ == '__main__':
    main()