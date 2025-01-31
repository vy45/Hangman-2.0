import subprocess
import sys
import time
import logging

def run_command(command):
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output but filter progress bars and frequent updates
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                # Only print lines that aren't progress bars or frequent updates
                if not any(x in line.lower() for x in 
                    ['|', '/', '\\', '[', ']', '%', 'it/s', 'calculating', 'simulating']):
                    print(line)
                
        # Wait for process to complete and get return code
        return_code = process.poll()
        return return_code == 0
        
    except Exception as e:
        print(f"Error running command: {str(e)}")
        return False

def main():
    # First run mary_hangman with force-new-data
    print("\n=== Running Mary Hangman ===")
    mary_result = run_command("python mary_hangman.py")
    print(f"Mary Hangman {'completed' if mary_result else 'failed'}")
    
    # Wait a moment between runs
    time.sleep(2)
    
    # Run transformer_hangman regardless of mary_hangman result
    print("\n=== Running Transformer Hangman ===")
    transformer_result = run_command("python transformer_hangman.py")
    print(f"Transformer Hangman {'completed' if transformer_result else 'failed'}")

if __name__ == "__main__":
    main() 