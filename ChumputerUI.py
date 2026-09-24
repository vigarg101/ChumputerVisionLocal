from flask import Flask, json, render_template, request, jsonify, send_file, redirect, url_for
import os
import ChumputerFunctions 
import traceback
import pandas as pd
import PIL.Image as Image
from pathlib import Path
import numpy as np  
import tempfile
import shutil
from werkzeug.utils import secure_filename   
from datetime import datetime, time as dtime
import requests
from collections import defaultdict
from apex_logger import ApexLogger
import time


app = Flask(__name__)

db_logger = ApexLogger("https://apex.psmfc.org/akfin/data_marts/akmp")

global_organization = "Unknown"
global_current_log_id = None
global_test_images_link = None
global_x_test = None
global_scratch_dir = None 
global_json_array = []
global_index_mapping = []
global_username = "VGarg_Test"
global_index_run = 0

model_path_dict = {}

# These will be for long format models
global_json_array_long = []
global_index_mapping_long = [] 

global_approval_rate = 0.0

global_run_start_time = None

# NEW: Offline mode flag
global_offline_mode = False


def fetch_users_from_apex():
    api_url = "https://apex.psmfc.org/akfin/data_marts/akmp/chumputer_users"
    try:
        response = requests.get(api_url, timeout=30)
        if response.status_code == 200:
            users = response.json().get('users', []) 
            frontend_users = []
            
            for user in users:
                username = user.get('username') or user.get('USERNAME')
                organization = user.get('organization') or user.get('ORGANIZATION') 
                email = user.get('email') or user.get('EMAIL') 
                if username:
                    frontend_users.append({"username": username, "organization": organization, "email": email})
            return frontend_users
        else:
            print(f"APEX Error fetching users: {response.status_code}")
            return []
    except Exception as e:
        traceback.print_exc()
        return []


print("Server starting: Loading models into VRAM. Please wait...")

try:
    predictor, target_feats, weights, weights_np, strategy = ChumputerFunctions.initialize_pipeline()
    print("Models loaded successfully!")
    
    print("Syncing latest Run ID from database...")
    try:
        hist_resp = requests.get("https://apex.psmfc.org/akfin/data_marts/akmp/chumputer_history", timeout=15)
        if hist_resp.status_code == 200:
            records = hist_resp.json().get('items', [])
            if records:
                global_index_run = max([int(r.get('run_id', 0)) for r in records if r.get('run_id') is not None], default=0)
        print(f"Run ID synced successfully. Next run will start at: {global_index_run + 1}")
    except Exception as e:
        print(f"Warning: Could not sync Run ID (likely offline). Error: {e}")

    print("Flask server is ready.")
except Exception as e:
    print(f"Failed to load models. Check GPU/Paths. Error: {e}")


def check_file_size(file_list, max_size):
    MAX_SIZE = max_size * 1024 * 1024  
    
    indexed_list = []
    current_batch = []
    current_batch_size = 0

    for file_pair in file_list:
        filepath = file_pair[0]
        masked_filepath = file_pair[1]
        
        size1 = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        size2 = os.path.getsize(masked_filepath) if os.path.exists(masked_filepath) else 0
        pair_size = size1 + size2

        if (current_batch_size + pair_size) > MAX_SIZE and current_batch:
            indexed_list.append(current_batch)
            current_batch = []
            current_batch_size = 0

        current_batch.append(file_pair)
        current_batch_size += pair_size

    if current_batch:
        indexed_list.append(current_batch)

    return indexed_list



@app.before_request
def enforce_mode_routing():
    global global_offline_mode, global_username, global_organization
    
    if request.endpoint == 'static':
        return

    raw_iap_email = request.headers.get('X-Goog-Authenticated-User-Email')

    if not raw_iap_email:

        global_offline_mode = True
        global_username = "Offline_Local_User"
        global_organization = "Local"
        
        offline_allowed_endpoints = [
            'upload_approve', 
            'model_run', 
            'handle_approval', 
            'save_approved', 
            'handle_run', 
            'get_models', 
            'serve_image'
        ]
        
        if request.endpoint and request.endpoint not in offline_allowed_endpoints:
            return redirect(url_for('upload_approve'))
    else:
        global_offline_mode = False


@app.route('/upload_approve')
def upload_approve():
    return render_template('upload_approve2.html')


@app.route('/model_run')
def model_run():
    return render_template('model_run.html')


@app.route('/history', methods=['GET', 'POST'])
def history_page():
    global global_username
    global global_organization 
    
    if request.method == 'POST':
        selected_user = request.form.get('username')
        
        if selected_user:
            global_username = selected_user
            global_organization = "Unknown" 
            
            all_users = fetch_users_from_apex()
            
            for user in all_users:
                if user['username'] == selected_user:
                    global_organization = user['organization'] or "Unknown"
                    break
                    
            print(f"Session updated. User proceeding as: {global_username} ({global_organization})")

    return render_template('history2.html', global_username=global_username, global_organization=global_organization)


@app.route('/')
def login():
    return render_template('login.html', global_username=global_username)


@app.route('/api/image')
def serve_image():
    """This route acts as a bridge to send local images to the browser securely."""
    image_path = request.args.get('path')
    
    if not image_path or not os.path.exists(image_path):
        return "Image not found", 404
        
    return send_file(image_path)


@app.route('/api/approve', methods=['POST'])
def handle_approval():

    global global_x_test, global_test_images_link, global_scratch_dir
    global global_index_mapping, global_json_array, global_username, global_json_array_long
    global global_current_log_id, global_organization, global_index_run

    is_last_batch = request.form.get('is_last_batch', 'false').lower() == 'true'
    batch_index = request.form.get('batch_index', None)

    if batch_index is not None:
        try:
            batch_index = int(batch_index)
            print(f"Processing batch index: {batch_index}, Is last batch: {is_last_batch}")
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid batch index."}), 400

    if batch_index == 0:    
        global_current_log_id = db_logger.start_run(
            username=global_username, 
            org=global_organization
        )

        db_logger.log(global_current_log_id, global_username, f"Initializing new run session for {global_organization} (Offline: {global_offline_mode})...")

        global_index_mapping.clear()
        global_json_array.clear()  
        global_json_array_long.clear()      
        global_index_mapping_long.clear() 

        if global_scratch_dir and os.path.exists(global_scratch_dir):
            try:
                shutil.rmtree(global_scratch_dir)
                print(f"Cleaned up old scratch directory: {global_scratch_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean up old scratch directory: {e}")

        global_scratch_dir = tempfile.mkdtemp(prefix="chumputer_batch_")
        db_logger.log(global_current_log_id, global_username, f"Created temporary remote upload directory: {global_scratch_dir}")

    if 'images' not in request.files:
        return jsonify({"status": "error", "message": "No files were sent in the request."}), 400

    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({"status": "error", "message": "No valid files were sent in the request."}), 400

    for file in files:
        if file.filename != '':
            filename = secure_filename(file.filename) 
            save_path = os.path.join(global_scratch_dir, filename)
            file.save(save_path)

    db_logger.log(global_current_log_id, global_username, f"Saved {len(files)} files from Batch {batch_index + 1}")

    if not is_last_batch:
        db_logger.log(global_current_log_id, global_username, f"Batch {batch_index + 1} uploaded successfully.")
        return jsonify({"status": "success", "message": f"Batch {batch_index + 1} uploaded successfully."}) 

    try:
        saved_files = os.listdir(global_scratch_dir)
        db_logger.log(global_current_log_id, global_username, f"DEBUG: Found {len(saved_files)} files physically sitting in {global_scratch_dir}")
        
        x_test, test_images_link, random_images_data, confidence_scores = ChumputerFunctions.process_images2(
            global_scratch_dir, predictor, target_feats, weights, weights_np
        )

        raw_start_time = request.form.get('start_time')
        if raw_start_time:
            start_seconds = float(raw_start_time) / 1000.0
            total_elapsed_time = time.time() - start_seconds
            print(f"\n==========================================")
            print(f"DEBUG: Total elapsed time for all batches: {total_elapsed_time:.2f} seconds.")
            print(f"==========================================")

        global_x_test = x_test
        global_test_images_link = test_images_link
        x = 0

        for img in random_images_data:
            tiff_img_path = img['filepath']
            filename = os.path.basename(tiff_img_path)

            temp_json = {
                "username": global_username,
                "file_name": filename,
                "date_time": datetime.now().isoformat()
            }

            global_index_mapping.append([filename, x])
            global_json_array.append(temp_json)
            x += 1
            
            tiff_image = Image.open(tiff_img_path)
            jpg_img_path = tiff_img_path.replace('.tif', '.jpg')
            jpeg_image = tiff_image.convert("RGB")
            my_file = Path(jpg_img_path)
            if not my_file.is_file():
                jpeg_image.save(jpg_img_path)
            img['filepath'] = jpg_img_path

        db_logger.log(global_current_log_id, global_username, f"Batch processing completed. Total images processed: {len(random_images_data)}")
        return jsonify({"status": "success", "results": random_images_data})

    except Exception as e:
        db_logger.log(global_current_log_id, global_username, f"Error during approval handling: {e}")
        traceback.print_exc() 
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/save_approved', methods=['POST'])
def save_approved():
    global global_x_test, global_test_images_link, global_current_log_id, global_username
    
    data = request.json
    approved_paths = data.get('approved_paths', [])

    if not approved_paths:
        db_logger.log(global_current_log_id, global_username, "No approved images provided for saving.")
        return jsonify({"status": "error", "message": "No approved images provided."}), 400

    try:
        normalized_links = [
            link['filepath'] if isinstance(link, dict) else link 
            for link in global_test_images_link
        ]
        
        approved_stems = {Path(p).stem for p in approved_paths}
        dataset_stems = [Path(path).stem for path in normalized_links]
        
        valid_indices = [
            i for i, path in enumerate(normalized_links) 
            if Path(path).stem in approved_stems
        ]
        
        if not valid_indices:
            db_logger.log(global_current_log_id, global_username, "Paths did not match dataset.")
            return jsonify({"status": "error", "message": "Paths did not match dataset."}), 400

        global_test_images_link = [global_test_images_link[i] for i in valid_indices]
        
        if isinstance(global_x_test, np.ndarray):
            global_x_test = global_x_test[valid_indices]
        elif isinstance(global_x_test, list):
            global_x_test = [global_x_test[i] for i in valid_indices]
            
        db_logger.log(global_current_log_id, global_username, f"Dataset updated. Retained {len(valid_indices)} approved images.")
        return jsonify({"status": "success"})
        
    except Exception as e:
        db_logger.log(global_current_log_id, global_username, f"Error filtering approved images: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/run_pipeline', methods=['POST'])
def handle_run():
    data = request.json
    selected_models = data.get('selectedModels')

    global global_x_test, global_test_images_link, global_index_mapping, global_json_array
    global global_index_run, global_json_array_long, global_scratch_dir
    global model_path_dict, global_current_log_id, global_username, global_offline_mode

    try:
        global_json_array_long.clear()
        db_logger.log(global_current_log_id, global_username, f"Starting prediction run with models: {selected_models}")
        
        model_predictions = ChumputerFunctions.run_predictions(
            selected_models, global_x_test, global_test_images_link, strategy, base_folder=model_path_dict
        )

        for idx in global_index_mapping:
            global_index_run += 1
            matching_rows = model_predictions.loc[model_predictions['filename'] == idx[0]]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                global_json_array[idx[1]]['weighted_average_predictions'] = float(row["combined_predictions"])
                global_json_array[idx[1]]['segmentation_approved'] = "Y" 
                global_json_array[idx[1]]['run_id'] = global_index_run

                for model in selected_models:
                    model_col = model
                    temp_json_long = {"run_id": global_index_run}
                    temp_json_long["model_name"] = model 
                    
                    if model_col in row:
                        temp_json_long["prediction"] = float(row[model_col])
                    else:
                        temp_json_long["prediction"] = None
                        
                    global_json_array_long.append(temp_json_long)
            else:
                global_json_array[idx[1]]['segmentation_approved'] = "N"
                global_json_array[idx[1]]['run_id'] = global_index_run
                temp_json_long = {
                    "run_id": global_index_run,
                    "model_name": None,
                    "prediction": None
                }
                global_json_array_long.append(temp_json_long)
            
        db_logger.log(global_current_log_id, global_username, "Final JSON array generated.")

        # ==========================================
        # BYPASS APEX DB SAVING IF IN OFFLINE MODE
        # ==========================================
        if not global_offline_mode:
            api_url = "https://apex.psmfc.org/akfin/data_marts/akmp/chumputer_history"
            x = requests.post(api_url, json=global_json_array, timeout=30)
            
            if x.status_code != 200:
                db_logger.log(global_current_log_id, global_username, f"Warning: Failed to save to database. Status Code: {x.status_code}")
                db_logger.log(global_current_log_id, global_username, f"APEX Error Message: {x.text}") 
            else:
                db_logger.log(global_current_log_id, global_username, "Successfully saved to APEX database!")

            api_url_long = "https://apex.psmfc.org/akfin/data_marts/akmp/chumputer_model_run" 
            x_long = requests.post(api_url_long, json=global_json_array_long, timeout=30)
            
            if x_long.status_code != 200:
                db_logger.log(global_current_log_id, global_username, f"Warning: Failed to save long format to database. Status Code: {x_long.status_code}")
            else:
                db_logger.log(global_current_log_id, global_username, "Successfully saved long format models to APEX!")
        else:
            db_logger.log(global_current_log_id, global_username, "Offline Mode: Bypassed saving run data to APEX webservices.")


        if isinstance(model_predictions, pd.DataFrame):
            html_content = model_predictions.to_html(
                classes="results-table", 
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        else:
            html_content = f"<pre>{model_predictions}</pre>"

        images_processed_count = len(global_index_mapping)
        db_logger.log(global_current_log_id, global_username, f"Run completed successfully. Processed {images_processed_count} images.")
        
        db_logger.end_run(
            log_id=global_current_log_id,
            completed='Y',
            images_processed=images_processed_count,
            username=global_username
        )
        global_current_log_id = None

        return jsonify({"status": "success", "results": html_content})

    except Exception as e:
        db_logger.log(global_current_log_id, global_username, "Error during pipeline execution:")
        traceback.print_exc() 
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Fetches history from APEX, filters by user, and aggregates into batches."""
    api_url = "https://apex.psmfc.org/akfin/data_marts/akmp/chumputer_history" 
    target_user = global_username

    try:
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            apex_data = response.json()
            records = apex_data.get('items', [])
            
            user_records = []
            for r in records:
                db_user = r.get('username') or r.get('USERNAME') or r.get('Username')
                if db_user and db_user.strip() == target_user:
                    user_records.append(r)
            
            total_historical_records = len(user_records)
            global_approval_count = sum(1 for r in user_records if r.get('segmentation_approved') == 'Y')
            global_approval_rate = (global_approval_count / total_historical_records) * 100 if total_historical_records > 0 else 0

            batches = defaultdict(list)
            for record in user_records:
                batch_time = record.get('date_ran') 
                if batch_time:
                    batches[batch_time].append(record)

            active_model_names = []
            try:
                models_url = "https://apex.psmfc.org/akfin/data_marts/akmp/chumputer_models"
                models_resp = requests.get(models_url, timeout=10)
                if models_resp.status_code == 200:
                    models_data = models_resp.json().get('models', [])
                    active_model_names = [m.get('model_name') for m in models_data if 'model_name' in m]
            except Exception as e:
                print(f"Warning: Could not fetch active models for stats: {e}")
        
            global_stats = {
                "total_historical_records": total_historical_records,
                "approval_rate": f"{global_approval_rate:.2f}%",
                "active_models": active_model_names
            }

            summary_data = []
            for timestamp, batch_records in batches.items():
                total_processed = len(batch_records)
                approved_count = sum(1 for r in batch_records if r.get('segmentation_approved') == 'Y')
                flagged_count = sum(1 for r in batch_records if r.get('segmentation_approved') == 'N')
                
                summary_data.append({
                    "batch_timestamp": timestamp,
                    "total_processed": total_processed,
                    "approved": approved_count,
                    "flagged": flagged_count,
                    "records": batch_records 
                })
            
            summary_data.sort(key=lambda x: x['batch_timestamp'], reverse=True)
            return jsonify({"status": "success", "batches": summary_data, "global_stats": global_stats})
            
        else:
            print(f"APEX Error: {response.status_code}")
            return jsonify({"status": "error", "message": "Failed to fetch from APEX"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    global global_scratch_dir, model_path_dict, global_offline_mode


    if global_offline_mode:

        offline_model_dict = {
            "ConvNeXt": "/opt/chumputer",
            "effnetv2l": "/opt/chumputer",
            "NASNetLarge": "/opt/chumputer"
        }
        
        model_names = []
        for name, base_folder in offline_model_dict.items():
            model_names.append(name)
            model_path_dict[name] = base_folder
            
        return jsonify({"status": "success", "models": model_names})


    api_url = "https://apex.psmfc.org/akfin/data_marts/akmp/chumputer_models"
    try:
        response = requests.get(api_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', []) 
            model_names = []
            
            for model in models:
                model_name = model.get('model_name') or model.get('MODEL_NAME')
                base_folder = model.get('base_folder_path') or model.get('BASE_FOLDER_PATH')
                
                if model_name:
                    model_names.append(model_name)
                    model_path_dict[model_name] = base_folder
                    
            return jsonify({"status": "success", "models": model_names})
        else:
            print(f"APEX Error: {response.status_code}")
            return jsonify({"status": "error", "message": "Failed to fetch models from APEX"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/users', methods=['GET'])
def get_users():
    """Fetches available users from the database for the frontend."""
    global global_offline_mode
    
    if global_offline_mode:
        return jsonify({"status": "success", "users": [{"username": "Offline_Local_User", "organization": "Local", "email": "offline@local"}]})

    users_list = fetch_users_from_apex()
    raw_iap_email = request.headers.get('X-Goog-Authenticated-User-Email')

    if not raw_iap_email:
        return jsonify({"status": "error", "message": "User email not found in request headers."}), 400

    cleaned_email = raw_iap_email.replace('accounts.google.com:', '').strip().lower()    

    matching_users = [
        user for user in users_list 
        if user.get('email') and user.get('email').lower() == cleaned_email
    ]

    if not matching_users:    
        return jsonify({"status": "error", "message": "User not found."}), 404

    if users_list:
        return jsonify({"status": "success", "users": matching_users})
    else:
        return jsonify({"status": "error", "message": "Failed to fetch users from APEX"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)












