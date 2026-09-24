import requests
import os
import json
import time
from datetime import datetime

class ApexLogger:
    def __init__(self, base_url, log_directory="chumputer_logs"):
        self.base_url = base_url
        self.log_directory = log_directory
        self.detail_logs = []
        self.local_mode = False  

        os.makedirs(self.log_directory, exist_ok=True)

    def start_run(self, username, org=None):
        """Creates the header record and returns the new ID."""

        self.detail_logs = []

        if self.local_mode:
            return int(time.time())

        url = f"{self.base_url}/chumputer_logs"
        payload = {
            "username": username,
            "org": org,
            "run_started": datetime.now().isoformat()
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("id") 
        except Exception as e:
            print(f"DB Logging Error (start_run): {e}")
        return None

    def log(self, log_id, username, message):
        """Prints to the console AND sends it to the detail table."""
        print(message) 
        
        if not log_id:
            return
            
        payload = {
            "log_id": log_id,
            "username": username,
            "message": str(message)[:4000], 
            "log_time": datetime.now().isoformat()
        }
        self.detail_logs.append(payload)

    def end_run(self, log_id, completed='Y', error_message=None, images_processed=0, username=None):
        """Updates the header record when the run finishes."""
        if not log_id:
            return

        master_log_file = os.path.join(self.log_directory, "chumputer_local_logs.txt")
        try:
            with open(master_log_file, 'a') as log_file:
                log_file.write(f"\n{'='*60}\n")
                log_file.write(f"RUN ID: {log_id} | STATUS: {'SUCCESS' if completed == 'Y' else 'FAILED'} | MODE: {'LOCAL' if self.local_mode else 'ONLINE'}\n")
                log_file.write(f"{'='*60}\n")
                
                for entry in self.detail_logs:
                    time_formatted = entry['log_time'].replace('T', ' ')
                    user = entry['username']
                    msg = entry['message']
                    
                    log_file.write(f"[{time_formatted}] [{user}] -> {msg}\n")
        except Exception as e:
            print(f"Failed to append to local log file: {e}")

        if self.local_mode:
            print("Local mode active: Skipping DB logging.")
            self.detail_logs = []
            return

        
        if completed == 'N':
            print(f"Run ended with errors")
            detail_url = f"{self.base_url}/chumputer_logs_detail"
            for detail_payload in self.detail_logs:
                try:
                    requests.post(detail_url, json=detail_payload, timeout=10)
                except Exception as e:
                    print(f"DB Logging Error (end_run detail): {e}")



        master_log_file = os.path.join(self.log_directory, "chumputer_master_history.log")
        try:
            # 'a' mode appends to the file instead of overriding it
            with open(master_log_file, 'a') as log_file:
                # Create a clean header for the run
                log_file.write(f"\n{'='*60}\n")
                log_file.write(f"RUN ID: {log_id} | STATUS: {'SUCCESS' if completed == 'Y' else 'FAILED'}\n")
                log_file.write(f"{'='*60}\n")
                
                # Parse the JSON array into human-readable lines
                for entry in self.detail_logs:
                    time_formatted = entry['log_time'].replace('T', ' ')
                    user = entry['username']
                    msg = entry['message']
                    
                    log_file.write(f"[{time_formatted}] [{user}] -> {msg}\n")
                    
        except Exception as e:
            print(f"Failed to append to master log file: {e}")


        url = f"{self.base_url}/chumputer_logs"
        payload = {
            "id": log_id,
            "run_ended": datetime.now().isoformat(),
            "completed": completed,
            "error_message": error_message,
            "images_processed": images_processed
        }
        try:
            requests.put(url, json=payload, timeout=10)
        except Exception as e:
            print(f"DB Logging Error (end_run): {e}")

        self.detail_logs = []