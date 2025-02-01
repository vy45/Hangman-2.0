import subprocess
import time
import logging
from datetime import datetime
import sys
from pathlib import Path

def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    Path(log_dir).mkdir(exist_ok=True)
    log_file = f'{log_dir}/run_models_v2_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def run_model(script_name, force_new_data=False, timeout=7200):  # 2 hour timeout
    cmd = ['python', script_name]
    if force_new_data:
        cmd.append('--force-new-data')
    
    logging.info(f"\nRunning {script_name}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time with timeout
        while True:
            if time.time() - start_time > timeout:
                process.kill()
                logging.error(f"{script_name} timed out after {timeout} seconds")
                return False
                
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logging.info(output.strip())
        
        # Get any remaining output
        _, stderr = process.communicate()
        if stderr:
            logging.error(stderr)
            
        if process.returncode != 0:
            logging.error(f"{script_name} failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"Error running {script_name}: {str(e)}")
        return False
        
    duration = time.time() - start_time
    logging.info(f"{script_name} completed in {duration:.2f} seconds")
    return True

def main():
    log_file = setup_logging()
    logging.info("Starting model training sequence")
    
    try:
        # Run NV3 with force new data
        if not run_model('mary_hangman_nv3.py'):
            raise Exception("NV3 training failed")
        
        # Run NV4 using NV3's data
        if not run_model('mary_hangman_nv4.py'):
            raise Exception("NV4 training failed")

        # Run NV5 using same data
        if not run_model('mary_hangman_nv5.py'):
            raise Exception("NV5 training failed")
        
        logging.info("\nAll models completed successfully!")
        
    except Exception as e:
        logging.error(f"Model sequence failed: {str(e)}")
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 