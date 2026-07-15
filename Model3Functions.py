import sys
print('DEBUG start: __name__ =', __name__)
# make Scalevision package importable
sys.path.insert(0, '/opt/chumputer')
print("sys.path after insert:", sys.path[:3])  # debug

from pathlib import Path
import os
# configure TensorFlow logging after importing os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import gc
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.nn import functional as F
from skimage.transform import resize as sk_resize
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ConvNeXtLarge
from tensorflow.keras.initializers import Constant
from tqdm import tqdm

from Scalevision.per_segment_anything import SamPredictor, sam_model_registry



def setup_gpu_memory_growth():
    """Enable memory growth for GPUs to prevent allocation issues."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


class Mask_Weights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

    def forward(self, x):
        return x * self.weights



def input_image_folder():
    """Prompt user for input image folder path."""
    while True:
        folder_path = input("Enter the path to the folder containing images: ")
        if os.path.isdir(folder_path):
            return folder_path
        else:
            print("Invalid folder path. Please try again.")



def load_sam_model(model_path: str, device: str = 'cuda') -> tuple:
    """Load SAM model components."""
    model_path = Path(model_path)
    file_names = ['mask_weights.pth', 'target_feats.pth', 'weights.pth']
    file_paths = {file_name: model_path / file_name for file_name in file_names}

    if not all(path.exists() for path in file_paths.values()):
        raise FileNotFoundError("Required SAM model files not found.")
    
    print('Loading existing model')
    # Load base SAM model
    sam_ckpt = model_path / 'sam_vit_h_4b8939.pth'
    sam = sam_model_registry['vit_h'](checkpoint=sam_ckpt).to(device)
    predictor = SamPredictor(sam)

    # Load mask weights
    mask_weights = Mask_Weights().to(device)
    mask_weights.load_state_dict(torch.load(file_paths['mask_weights.pth'], map_location=device))

    # Load target features and weights
    target_feats = torch.load(file_paths['target_feats.pth'], map_location=device)
    weights = torch.load(file_paths['weights.pth'], map_location=device)
    weights_np = weights.detach().cpu().numpy()

    return predictor, mask_weights, target_feats, weights, weights_np



def get_bounding_box(binary_mask: np.ndarray) -> tuple:
    """Calculate bounding box from binary mask."""
    y_coords, x_coords = np.where(binary_mask)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return 0, 0, 0, 0
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def crop_to_mask(input_image: Image, binary_mask: np.ndarray) -> Image:
    """Crop image to mask bounding box."""
    mask_image = Image.fromarray(binary_mask)
    bounding_box = get_bounding_box(binary_mask)

    cropped_input = input_image.crop(bounding_box)
    cropped_mask = mask_image.crop(bounding_box)

    grayscale_mask = cropped_mask.convert('L')
    background = Image.new('L', cropped_input.size)

    masked_result = Image.composite(cropped_input.convert('L'), background, grayscale_mask)
    return masked_result.convert('RGB')


def point_selection(mask_sim: torch.Tensor, topk: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Select top-k points from mask similarity."""
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    return topk_xy.cpu().numpy(), topk_label


def get_image_mask(image_blob: bytes, predictor, target_feats, weights, weights_np) -> np.ndarray:
    """Generate mask using SAM."""
    size = 1024
    test_image = cv2.imdecode(np.frombuffer(image_blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (size, size))

    predictor.set_image(test_image)
    test_feat = predictor.features.squeeze()

    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(C, h * w)

    k = 5
    sim_list = [targ_feat @ test_feat for targ_feat in target_feats]
    top_k_sim_list = []
    for sim in sim_list:
        top_k_values, _ = torch.topk(sim.view(-1), k)
        sim[sim < top_k_values[-1]] = 0
        top_k_sim_list.append(sim)
    sim = sum(top_k_sim_list) / len(top_k_sim_list)

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode='bilinear')
    sim = predictor.model.postprocess_masks(sim, input_size=predictor.input_size, original_size=predictor.original_size).squeeze()

    topk_xy, topk_label = point_selection(sim, topk=1)

    masks, scores, logits, logits_high_list = predictor.predict(
        point_coords=topk_xy, point_labels=topk_label, multimask_output=True
    )

    weights_list = [logit_high * weight.unsqueeze(-1) for logit_high, weight in zip(logits_high_list, weights)]
    logit_high = sum(weights_list) / len(weights_list)
    mask = (logit_high > 0).detach().cpu().numpy()

    logits_list = [logit * weight_np[..., None] for logit, weight_np in zip(logits, weights_np)]
    logit = sum(logits_list) / len(logits_list)

    y, x = np.nonzero(mask)
    if y.size == 0 or x.size == 0:
        print('Mask has no non-zero values.')
        return mask
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy, point_labels=topk_label, box=input_box[None, :], mask_input=logit[None, :, :], multimask_output=True
    )
    best_idx = np.argmax(scores)

    y, x = np.nonzero(masks[best_idx])
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy, point_labels=topk_label, box=input_box[None, :], mask_input=logits[best_idx: best_idx + 1, :, :], multimask_output=True
    )
    best_idx = np.argmax(scores)

    return masks[best_idx], scores[best_idx]


def image_to_bytes(image: Image) -> bytes:
    """Convert PIL Image to bytes."""
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()



def process_image(original_image: Image, predictor, target_feats, weights, weights_np) -> Image:
    """Process a single image with SAM and crop to mask.
    
    Args:
        original_image: PIL Image to process
        predictor: SAM predictor object
        target_feats: SAM target features
        weights: SAM weights
        weights_np: SAM weights as numpy array
        
    Returns:
        Processed PIL Image
    """
    image_bytes = image_to_bytes(original_image)

    mask: np.ndarray

    mask, confidence_score = get_image_mask(image_bytes, predictor, target_feats, weights, weights_np)

    resized_mask: np.ndarray = sk_resize(
        mask,
        (original_image.size[1], original_image.size[0]),
        anti_aliasing=False,
        mode='edge',
    )

    masked_image: Image = crop_to_mask(original_image, resized_mask)
    return masked_image, confidence_score

def classify_and_store_images(base_folder: str, predictor, target_feats, weights, weights_np):
    """Process all images in the specified folder with SAM.
    
    Args:
        base_folder: Path to the directory containing images to process
        predictor: SAM predictor object
        target_feats: SAM target features
        weights: SAM weights
        weights_np: SAM weights as numpy array
        
    Returns:
        List of tuples (filename, processed_image)
    """
    sam_images_in_memory = []

    for file_name in os.listdir(base_folder):
        if file_name.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg')):
            image_path = os.path.join(base_folder, file_name)
            try:
                with Image.open(image_path) as img:
                    processed_image, confidence_score = process_image(img, predictor, target_feats, weights, weights_np)
                    sam_images_in_memory.append((file_name, processed_image, confidence_score))

                    print(f"Processed {file_name} and stored in memory")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    return sam_images_in_memory

import os
import PIL.Image as Image

def classify_and_store_images_2(base_folder: str, predictor, target_feats, weights, weights_np):
    """
    Process all raw images in the specified folder with SAM, 
    while safely ignoring previously generated output files.
    """
    sam_images_in_memory = []

    for file_name in os.listdir(base_folder):
        lower_name = file_name.lower()
        
        if lower_name.endswith('_masked.jpg'):
            continue
            
        if lower_name.endswith(('.jpg', '.jpeg')):
            base_name = os.path.splitext(file_name)[0]
            if os.path.exists(os.path.join(base_folder, f"{base_name}.tif")) or \
               os.path.exists(os.path.join(base_folder, f"{base_name}.tiff")):
                continue 

        if lower_name.endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg')):
            image_path = os.path.join(base_folder, file_name)
            try:
                with Image.open(image_path) as img:
                    processed_image, confidence_score = process_image(img, predictor, target_feats, weights, weights_np)
                    sam_images_in_memory.append((file_name, processed_image, confidence_score))

                    print(f"Processed {file_name} and stored in memory")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    return sam_images_in_memory



def resize_images_in_place(sam_images_list, target_size=(512, 512)):
    """Resize images in place and replace in list to save memory."""
    for i in range(len(sam_images_list)):
        fname, img = sam_images_list[i]
        if img.size != target_size:
            resized_img = img.resize(target_size, Image.LANCZOS)
            img.close()
            sam_images_list[i] = (fname, resized_img)
            del img
    gc.collect()


def load_images_from_memory(image_filenames: list, sam_images_list: list) -> dict:
    """Load images from in-memory list by filename.
    
    Args:
        image_filenames: List of filenames to retrieve
        sam_images_list: List of tuples (filename, PIL.Image)
        
    Returns:
        Dict mapping filename to PIL.Image
    """
    sam_dict = dict(sam_images_list)
    images = {}
    for filename in image_filenames:
        img = sam_dict.get(filename)
        if img is not None:
            images[filename] = img
        else:
            print(f"{filename} not found in in-memory image list.")
    return images


    
def process_image_for_cnn(img):

    if not isinstance(img, Image.Image):
        raise TypeError("Input has to be a PIL.Image.Image object.")
    
    img_np = np.array(img)

    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    elif len(img_np.shape) == 3:
        if img_np.shape[2] == 1:
            img_rgb = np.repeat(img_np, 3, axis=2)

        elif img_np.shape[2] == 3:
            img_rgb = img_np

        else:
            raise ValueError(f"Unexpected number of channels: {img_np.shape[2]}")

    else:
        raise ValueError(f"Unexpected image dimensions: {img_np.shape}")
    
    return img_rgb


def process_images_for_cnn(image_dict):
    """Process images for CNN input.
    
    Args:
        image_dict: Dict mapping filename to PIL.Image
        
    Returns:
        Tuple of (processed_images_list, successful_files_list)
    """
    processed_images = []
    successful_files = []
    error_count = 0

    for filename, img in tqdm(image_dict.items(), desc="Processing Images"):
        try:
            processed_img = process_image_for_cnn(img)
            processed_images.append(processed_img)
            successful_files.append(filename)
        except (TypeError, ValueError) as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1

    print(f"Number of errors: {error_count}")
    return processed_images, successful_files


#####################################################################################################


class LayerScale(tf.keras.layers.Layer):
    """Layer scaling layer for ConvNeXt."""
    def __init__(self, init_values=1e-6, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        if self.projection_dim is None:
            if input_shape[-1] is None:
                raise ValueError("projection_dim must be set or inferable from input shape.")
            self.projection_dim = input_shape[-1]
        self.scale = self.add_weight(
            name='scale', shape=(self.projection_dim,), initializer=Constant(self.init_values), trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({'init_values': self.init_values, 'projection_dim': self.projection_dim})
        return config


def build_convnext_model(input_shape=(512, 512, 3), dropout_rate=0.5, l2_reg=1e-4) -> Model:
    """Build ConvNeXt model for age prediction."""
    pretrained_ConvNeXtLarge = ConvNeXtLarge(include_top=False, include_preprocessing=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(pretrained_ConvNeXtLarge.output)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='linear')(x)
    return Model(inputs=pretrained_ConvNeXtLarge.input, outputs=predictions)


def load_convnext_model(weights_path: str, strategy) -> Model:
    """Load ConvNeXt model with weights."""
    with strategy.scope():
        model = build_convnext_model()
        model.load_weights(weights_path)
    return model


def load_effnetv2l_model(weights_path: str, strategy) -> Model:
    """Load EfficientNetV2L model."""
    with strategy.scope():
        return load_model(weights_path)


def load_nasnet_model(weights_path: str, strategy) -> Model:
    """Load NASNet model."""
    with strategy.scope():
        return load_model(weights_path)


def predict_with_model(model, x_data: np.ndarray) -> np.ndarray:
    """Run predictions with a model."""
    return model.predict(x_data)





def load_model_histories(history_paths: dict) -> dict:
    """Load training histories from pickle files."""
    histories = {}
    for name, path in history_paths.items():
        with open(path, 'rb') as f:
            histories[name] = pickle.load(f)
    return histories


def calculate_ensemble_weights(histories: dict, min_mses: dict) -> np.ndarray:
    """Calculate weights for ensembling based on MSE."""
    mses = [min_mses[name] for name in ['effnet', 'nasnet', 'convnext']]
    weights = np.array([1/mse for mse in mses])
    return weights / weights.sum()


def ensemble_predictions(predictions_list: list, weights: np.ndarray) -> tuple:
    """Ensemble predictions from multiple models.
    
    Args:
        predictions_list: List of prediction arrays
        weights: Weight for each model
        
    Returns:
        Tuple of (combined_predictions, rounded_predictions)
    """
    predictions_adjusted = [pred + 1 for pred in predictions_list]
    combined = sum(w * pred for w, pred in zip(weights, predictions_adjusted))
    return combined, np.round(combined)


def create_predictions_dataframe(predictions_list: list, combined_preds: np.ndarray, filenames: list) -> pd.DataFrame:
    """Create DataFrame with predictions.
    
    Args:
        predictions_list: List of individual model predictions
        combined_preds: Combined ensemble predictions
        filenames: List of image filenames
        
    Returns:
        DataFrame with predictions and filenames
    """
    pred_arrays = (
        [combined_preds.reshape(-1, 1)] +
        [pred.reshape(-1, 1) for pred in predictions_list] +
        [np.round(combined_preds).reshape(-1, 1)]
    )
    pred = np.concatenate(pred_arrays, axis=1)
    df = pd.DataFrame(
        data=pred,
        columns=['combined_predictions', 'effnetv2l', 'NASNetLarge', 'ConvNeXt', 'combined_test_predictions_round']
    )
    df["filename"] = filenames
    return df


def save_predictions_to_csv(df: pd.DataFrame, output_path: str):
    """Save predictions to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")



####################################################################################################

def run_selected_models(selected_keys: list, x_test: np.ndarray, strategy) -> dict:
    """Run predictions for selected models.
    
    Args:
        selected_keys: List of model keys to run (e.g., ['convnext', 'effnet'])
        x_test: Test data for predictions
        strategy: TensorFlow distribution strategy"""
    

    results = {}
    weights = []

    if 'convnext' in selected_keys:
        convnext_model = load_convnext_model("/opt/chumputer/ConvNeXtLarge_SAM_model_file.hdf5", strategy)
        results['convnext'] = predict_with_model(convnext_model, x_test)
        with open('/opt/chumputer/ConvNeXtLargeSAM_SGD_history.pkl', 'rb') as f:
            ConvNeXt_history = pickle.load(f)
        ConvNeXt_test_mse = ConvNeXt_history['val_mse']
        ConvNeXt_min_val_mse = min(ConvNeXt_test_mse)

        mse3 = ConvNeXt_min_val_mse
        weights.append(1/mse3)

        
    if 'effnet' in selected_keys:
        effnet_model = load_effnetv2l_model("/opt/chumputer/effnetv2l_SAM_model_file.hdf5", strategy)
        results['effnet'] = predict_with_model(effnet_model, x_test)
        with open('/opt/chumputer/effnetv2lhistory_2.pkl', 'rb') as f:
            effnet_history = pickle.load(f)
        effnet_test_mse = effnet_history['val_mse']        
        effnet_min_val_mse = min(effnet_test_mse)

        mse1 = effnet_min_val_mse
        weights.append(1/mse1)


    if 'nasnet' in selected_keys:
        nasnet_model = load_nasnet_model("/opt/chumputer/NASNetLarge_SAM_model_file.hdf5", strategy)
        results['nasnet'] = predict_with_model(nasnet_model, x_test)
        #with open('/opt/chumputer/NASNetLarge_history.pkl', 'rb') as f:
            #NASNet_history = pickle.load(f)
        nasnet_min_val_mse = 0.0865

        mse2 = nasnet_min_val_mse
        weights.append(1/mse2)

    
    weights = np.array(weights)
    weights = weights / weights.sum()

    test_predictions1 = results['effnet'] + 1
    test_predictions2 = results['nasnet'] + 1
    test_predictions3 = results['convnext'] + 1

    predictions_round1 = np.round(test_predictions1)
    predictions_round2 = np.round(test_predictions2)
    predictions_round3 = np.round(test_predictions3)

    combined_predictions = weights[0] * test_predictions1 + weights[1] * test_predictions2 + weights[2] * test_predictions3
    combined_predictions_round = np.round(combined_predictions)

    combined_test_predictions = np.array(combined_test_predictions)
    combined_test_predictions_round = np.array(combined_test_predictions_round)
    test_predictions1 = np.array(test_predictions1)
    test_predictions2 = np.array(test_predictions2)
    test_predictions3 = np.array(test_predictions3)

    pred = np.concatenate((combined_test_predictions.reshape(-1, 1), test_predictions1.reshape(-1, 1), 
                       test_predictions2.reshape(-1, 1), test_predictions3.reshape(-1, 1), combined_test_predictions_round.reshape(-1, 1)), axis=1)  

    pred_df = pd.DataFrame(data=pred, columns=['combined_predictions', 'effnetv2l', 'NASNetLarge', 'ConvNeXt', 'combined_test_predictions_round']) 

    return pred_df



def run_selected_models2(selected_keys: list, x_test: np.ndarray, strategy) -> pd.DataFrame:
    """Run predictions for selected models.
    
    Args:
        selected_keys: List of model keys to run (e.g., ['convnext', 'effnet'])
        x_test: Test data for predictions
        strategy: TensorFlow distribution strategy
    """
    
    predictions = {}
    raw_weights = {}

    if 'ConvNeXt' in selected_keys:
        convnext_model = load_convnext_model("/opt/chumputer/ConvNeXtLarge_SAM_model_file.hdf5", strategy)
        predictions['ConvNeXt'] = predict_with_model(convnext_model, x_test) + 1
        
        with open('/opt/chumputer/ConvNeXtLargeSAM_SGD_history.pkl', 'rb') as f:
            ConvNeXt_history = pickle.load(f)
        raw_weights['ConvNeXt'] = 1 / min(ConvNeXt_history['val_mse'])

        
    if 'effnetv2l' in selected_keys:
        effnet_model = load_effnetv2l_model("/opt/chumputer/effnetv2l_SAM_model_file.hdf5", strategy)
        predictions['effnetv2l'] = predict_with_model(effnet_model, x_test) + 1
        
        with open('/opt/chumputer/effnetv2lhistory_2.pkl', 'rb') as f:
            effnet_history = pickle.load(f)
        raw_weights['effnetv2l'] = 1 / min(effnet_history['val_mse'])


    if 'NASNetLarge' in selected_keys:
        nasnet_model = load_nasnet_model("/opt/chumputer/NASNetLarge_SAM_model_file.hdf5", strategy)
        predictions['NASNetLarge'] = predict_with_model(nasnet_model, x_test) + 1
        
        raw_weights['NASNetLarge'] = 1 / 0.0865 

    
    if not predictions:
        return pd.DataFrame()

    total_weight = sum(raw_weights.values())
    normalized_weights = {model: weight / total_weight for model, weight in raw_weights.items()}

    first_model_key = list(predictions.keys())[0]
    combined_predictions = np.zeros_like(predictions[first_model_key])

    for model_name, pred in predictions.items():
        combined_predictions += pred * normalized_weights[model_name]

    df_data = {
        'combined_predictions': combined_predictions.flatten()
    }
    
    for model_name, pred in predictions.items():
        df_data[model_name] = pred.flatten()
        
    df_data['combined_test_predictions_round'] = np.round(combined_predictions).flatten()

    return pd.DataFrame(df_data)


def choose_models():

    x = 1
    while x == 1:
        print("Available models: 'convnext', 'effnet', 'nasnet'")
        selected_keys = input("Enter the models you want to run (comma-separated): ").split(',')
        selected_keys = [key.strip() for key in selected_keys]

        valid_keys = {'convnext', 'effnet', 'nasnet'}
        if all(key in valid_keys for key in selected_keys):
            return selected_keys
        else:
            print("Invalid selection. Please choose from 'convnext', 'effnet', 'nasnet'.")



def initialize_pipeline(sam_path='/opt/chumputer/Scalevision/crop_vision'):
    """Sets up GPU, loads the SAM model, and defines the TF strategy."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  

    setup_gpu_memory_growth()
    
    print("Loading SAM model...")
    predictor, mask_weights, target_feats, weights, weights_np = load_sam_model(sam_path)
    
    strategy = tf.distribute.MirroredStrategy()
    
    return predictor, target_feats, weights, weights_np, strategy





def select_approval(successful_files, base_folder):

    random_indices = np.random.choice(len(successful_files), size=min(5, len(successful_files)), replace=False)
    
    random_images_data = []
    for idx in random_indices:
        filename = successful_files[idx]
        full_path = os.path.join(base_folder, filename)
        
        random_images_data.append({"filepath": full_path})
        
    return random_images_data

def select_approval_2(successful_files, base_folder, sam_images):
    random_indices = np.random.choice(len(successful_files), size=min(5, len(successful_files)), replace=False)
    
    sam_dict = {fname: img for fname, img in sam_images}
    
    random_images_data = []
    
    for idx in range(len(successful_files)):
        filename = successful_files[idx]
        full_path = os.path.join(base_folder, filename)
        
        name, _ = os.path.splitext(filename)
        masked_filename = f"{name}_masked.jpg" 
        masked_full_path = os.path.join(base_folder, masked_filename)
        
        if filename in sam_dict:
            processed_img = sam_dict[filename]
            
            try:
                if hasattr(processed_img, 'save'):
                    processed_img.convert('RGB').save(masked_full_path)
                else:
                    Image.fromarray(processed_img).convert('RGB').save(masked_full_path)
            except Exception as e:
                print(f"Could not save masked image for {filename}: {e}")
        
        random_images_data.append({
            "filepath": full_path,
            "masked_filepath": masked_full_path
        })
        
    return random_images_data


def initialize_pipeline(sam_path='/opt/chumputer/Scalevision/crop_vision'):
    """Sets up GPU, loads the SAM model, and defines the TF strategy."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  

    setup_gpu_memory_growth()
    
    print("Loading SAM model...")
    predictor, mask_weights, target_feats, weights, weights_np = load_sam_model(sam_path)
    
    strategy = tf.distribute.MirroredStrategy()
    
    return predictor, target_feats, weights, weights_np, strategy


def process_images(base_folder, predictor, target_feats, weights, weights_np):
    """Classifies, crops, and processes images into a CNN-ready tensor."""
    print(f"Processing images from: {base_folder}")
    
    sam_images = classify_and_store_images_2(base_folder, predictor, target_feats, weights, weights_np)

    resize_images_in_place(sam_images, target_size=(512, 512))
    test_images_link = [fname for fname, _ in sam_images]
    test_images = load_images_from_memory(test_images_link, sam_images)

    processed_test_images, successful_files = process_images_for_cnn(test_images)
    #approve_images_for_cnn(processed_test_images, successful_files)

    random_images_data = select_approval_2(successful_files, base_folder, sam_images)

    x_test = np.array(processed_test_images).astype("float32") / 255
    
    return x_test, test_images_link, random_images_data


def process_images2(base_folder, predictor, target_feats, weights, weights_np):
    """Classifies, crops, and processes images into a CNN-ready tensor."""
    print(f"Processing images from: {base_folder}")
    
    sam_images_raw = classify_and_store_images_2(base_folder, predictor, target_feats, weights, weights_np)

    confidence_scores = {fname: float(score) for fname, _, score in sam_images_raw}

    sam_images = [(fname, img) for fname, img, _ in sam_images_raw]

    
    resize_images_in_place(sam_images, target_size=(512, 512))
    
    test_images_link = [fname for fname, _ in sam_images]
    test_images = load_images_from_memory(test_images_link, sam_images)

    processed_test_images, successful_files = process_images_for_cnn(test_images)

    random_images_data = select_approval_2(successful_files, base_folder, sam_images)

    x_test = np.array(processed_test_images).astype("float32") / 255
    
    return x_test, test_images_link, random_images_data, confidence_scores


def run_predictions(selected_keys, x_test, test_images_link, strategy):
    """Runs the selected models across the processed images using the mirrored strategy."""
    print(f"Running predictions for models: {selected_keys}")
    
    model_predictions = run_selected_models2(selected_keys, x_test, strategy)
    
    model_predictions['filename'] = test_images_link    
    
    return model_predictions





'''
if __name__ == "__main__":

    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logging
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Use PCI order for GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Make sure both GPUs are visible

    # Setup GPU and load SAM model
    setup_gpu_memory_growth()
    sam_path = '/opt/chumputer/Scalevision/crop_vision'
    predictor, mask_weights, target_feats, weights, weights_np = load_sam_model(sam_path)
    
    # Process images from the specified folder
    #base_folder = '/opt/chumputer/scale_images/subset_images'
    base_folder = input_image_folder()
    sam_images = classify_and_store_images(base_folder, predictor, target_feats, weights, weights_np)

    resize_images_in_place(sam_images, target_size=(512, 512))
    test_images_link = [fname for fname, _ in sam_images]
    test_images = load_images_from_memory(test_images_link, sam_images)

    processed_test_images, successful_files = process_images_for_cnn(test_images)

    approve_images_for_cnn(processed_test_images, successful_files)

    #x_test = processed_test_images.astype("float32") / 255
    x_test = np.array(processed_test_images).astype("float32") / 255

    strategy = tf.distribute.MirroredStrategy()
    #selected_keys = ['convnext']  # Example
    selected_keys = choose_models()
    #print (f"Selected models: {selected_keys}")
    model_predictions = run_selected_models2(selected_keys, x_test, strategy)
    model_predictions['filename'] = test_images_link    
    print(model_predictions)



    #print("hello world")'''