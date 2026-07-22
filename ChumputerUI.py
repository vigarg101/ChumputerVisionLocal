from flask import Flask, json, render_template, request, jsonify, send_file
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
from datetime import datetime
import requests
from collections import defaultdict



app = Flask(__name__)

global_test_images_link = None
global_x_test = None
global_scratch_dir = None 
global_json_array = []
global_index_mapping = []
global_username = "VGarg_Test"
global_index_run = 0

model_path_dict = {}

#These will be for long format models
global_json_array_long = []
global_index_mapping_long = [] 

global_approval_rate = 0.0
 





print("Server starting: Loading models into VRAM. Please wait...")

try:
    predictor, target_feats, weights, weights_np, strategy = ChumputerFunctions.initialize_pipeline()
    print("Models loaded successfully!")
    
    print("Syncing latest Run ID from database...")
    try:
        hist_resp = requests.get("https://apex.psmfc.org/pacfin/data_marts/pt/chumputer_history", timeout=15)
        if hist_resp.status_code == 200:
            records = hist_resp.json().get('items', [])
            if records:
                global_index_run = max([int(r.get('run_id', 0)) for r in records if r.get('run_id') is not None], default=0)
        print(f"Run ID synced successfully. Next run will start at: {global_index_run + 1}")
    except Exception as e:
        print(f"Warning: Could not sync Run ID. Error: {e}")

    print("Flask server is ready.")
except Exception as e:
    print(f"Failed to load models. Check GPU/Paths. Error: {e}")



@app.route('/upload_approve')
def upload_approve():
    return render_template('upload_approve.html')


@app.route('/model_run')
def model_run():
    return render_template('model_run.html')


@app.route('/')
def history_page():
    return render_template('history2.html', global_username=global_username)


@app.route('/api/image')
def serve_image():
    """This route acts as a bridge to send local images to the browser securely."""
    image_path = request.args.get('path')
    
    if not image_path or not os.path.exists(image_path):
        return "Image not found", 404
        
    return send_file(image_path)




@app.route('/api/approve', methods=['POST'])
def handle_approval():
    global global_x_test, global_test_images_link, global_scratch_dir, global_index_mapping, global_json_array, global_username, global_json_array_long

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

    if 'images' not in request.files:
        return jsonify({"status": "error", "message": "No files were sent in the request."}), 400

    files = request.files.getlist('images')

    if not files or files[0].filename == '':
        return jsonify({"status": "error", "message": "No files selected."}), 400

    try:
        global_scratch_dir = tempfile.mkdtemp(prefix="chumputer_batch_")
        print(f"Created temporary remote upload directory: {global_scratch_dir}")

        for file in files:
            filename = secure_filename(file.filename) 
            save_path = os.path.join(global_scratch_dir, filename)
            file.save(save_path)

        print(f"Successfully uploaded and saved {len(files)} files to the remote server.")

        saved_files = os.listdir(global_scratch_dir)
        print(f"DEBUG: Found {len(saved_files)} files physically sitting in {global_scratch_dir}")
        print(f"DEBUG: First 5 files look like this: {saved_files[:5]}")

        x_test, test_images_link, random_images_data, confidence_scores = ChumputerFunctions.process_images2(
            global_scratch_dir, predictor, target_feats, weights, weights_np
        )
        
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

        return jsonify({"status": "success", "results": random_images_data})

    except Exception as e:
        print("Error during approval handling:")
        traceback.print_exc() 
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/api/save_approved', methods=['POST'])
def save_approved():
    global global_x_test, global_test_images_link
    
    data = request.json
    approved_paths = data.get('approved_paths', [])

    if not approved_paths:
        return jsonify({"status": "error", "message": "No approved images provided."}), 400

    try:
        normalized_links = [
            link['filepath'] if isinstance(link, dict) else link 
            for link in global_test_images_link
        ]
        

        approved_stems = {Path(p).stem for p in approved_paths}
        
        valid_indices = [
            i for i, path in enumerate(normalized_links) 
            if Path(path).stem in approved_stems
        ]
        
        if not valid_indices:
            return jsonify({"status": "error", "message": "Paths did not match dataset."}), 400

        global_test_images_link = [global_test_images_link[i] for i in valid_indices]
        
        if isinstance(global_x_test, np.ndarray):
            global_x_test = global_x_test[valid_indices]
        elif isinstance(global_x_test, list):
            global_x_test = [global_x_test[i] for i in valid_indices]
            
        print(f"Dataset updated. Retained {len(valid_indices)} approved images.")
        return jsonify({"status": "success"})
        
    except Exception as e:
        print("Error filtering approved images:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/api/run_pipeline', methods=['POST'])
def handle_run():
    data = request.json
    selected_models = data.get('selectedModels')


    global global_x_test, global_test_images_link, global_index_mapping, global_json_array, global_index_run, global_json_array_long, global_scratch_dir

    global model_path_dict

    try:

        global_json_array_long.clear()

        print(f"Running predictions with: {selected_models}")
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

                    if model_col in row:
                        temp_json_long = {
                            "run_id": global_index_run
                        }
                        temp_json_long["model_name"] = model 
                        temp_json_long["prediction"] = float(row[model_col])
                        global_json_array_long.append(temp_json_long)

                    else:
                        temp_json_long = {
                            "run_id": global_index_run
                        }
                        temp_json_long["model_name"] = model
                        temp_json_long["prediction"] = None
                        global_json_array_long.append(temp_json_long)
                

            else:
                global_json_array[idx[1]]['segmentation_approved'] = "N"
                global_json_array[idx[1]]['run_id'] = global_index_run

                temp_json_long = {
                    "run_id": global_index_run
                }
                temp_json_long["model_name"] = None
                temp_json_long["prediction"] = None
                global_json_array_long.append(temp_json_long)

            
    
            

            
            
        print("\n=== FINAL JSON ARRAY ===")
        print(json.dumps(global_json_array, indent=4))
        print("========================\n")

        print("\n=== FINAL LONG FORMAT JSON ARRAY ===")
        print(json.dumps(global_json_array_long, indent=4))
        print("====================================\n")

        api_url = "https://apex.psmfc.org/pacfin/data_marts/pt/chumputer_history"
        x = requests.post(api_url, json=global_json_array, timeout=30)
        
        if x.status_code != 200:
            print(f"Warning: Failed to save to database. Status Code: {x.status_code}")
            print(f"APEX Error Message: {x.text}") 
        else:
            print("Successfully saved to APEX database!")
            print(f"APEX Server says: {x.text}")


        api_url_long = "https://apex.psmfc.org/pacfin/data_marts/pt/chumputer_model_run" 
        x_long = requests.post(api_url_long, json=global_json_array_long, timeout=30)
        
        if x_long.status_code != 200:
            print(f"Warning: Failed to save long format to database. Status Code: {x_long.status_code}")
            print(f"APEX Error Message: {x_long.text}")
        else:
            print("Successfully saved long format models to APEX!")


        if isinstance(model_predictions, pd.DataFrame):
            html_content = model_predictions.to_html(
                classes="results-table", 
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        else:
            html_content = f"<pre>{model_predictions}</pre>"

        return jsonify({"status": "success", "results": html_content})

    except Exception as e:
        print("Error during pipeline execution:")
        traceback.print_exc() 
        return jsonify({"status": "error", "message": str(e)}), 500





@app.route('/api/history', methods=['GET'])
def get_history():
    """Fetches history from APEX, filters by user, and aggregates into batches."""
    api_url = "https://apex.psmfc.org/pacfin/data_marts/pt/chumputer_history" 
    
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

            total_historical_records = len(user_records)

            
            batches = defaultdict(list)
            for record in user_records:
                batch_time = record.get('date_ran') 
                if batch_time:
                    batches[batch_time].append(record)


            active_model_names = []
            try:
                models_url = "https://apex.psmfc.org/pacfin/data_marts/pt/chumputer_models"
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


    global global_scratch_dir, model_path_dict


    "Fetches available models from the database"
    api_url = "https://apex.psmfc.org/pacfin/data_marts/pt/chumputer_models"

    try:
        response = requests.get(api_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            model_names = [model.get('model_name') for model in models if 'model_name' in model]
            model_path_dict.update({model_name: "/opt/chumputer" for model_name in model_names})
            return jsonify({"status": "success", "models": model_names})
        else:
            print(f"APEX Error: {response.status_code}")
            return jsonify({"status": "error", "message": "Failed to fetch models from APEX"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)