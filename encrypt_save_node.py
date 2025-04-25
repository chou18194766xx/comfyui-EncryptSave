# encrypt_save_node.py

import torch
import numpy as np
from PIL import Image
import io # Needed for saving PIL image to bytes
import json
import os
import time # Keep for potential timing/debugging
import folder_paths  # ComfyUI's path utility

# --- AES Encryption Dependencies ---
# Make sure pycryptodome is installed (pip install pycryptodome)
# Add 'pycryptodome' to requirements.txt in your custom node's directory
try:
    from Crypto.Cipher import AES
    from Crypto.Protocol.KDF import PBKDF2
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad, unpad
    from Crypto.Hash import SHA256
    PYCRYPTODOME_AVAILABLE = True
except ImportError:
    PYCRYPTODOME_AVAILABLE = False
    print("Warning: pycryptodome library not found. AES encryption node will not work.")
    print("Please install it: pip install pycryptodome")
    # Optionally, add instructions to add it to requirements.txt for ComfyUI Manager

# --- AES Configuration (matches the reference script) ---
SALT_SIZE = 16
KEY_SIZE = 32  # AES-256
PBKDF2_ITERATIONS = 100000
AES_BLOCK_SIZE = AES.block_size # Should be 16 for AES

# --- AES Encryption Function (operates on bytes) ---

def encrypt_data(plaintext_bytes, password):
    """
    Encrypts byte data using AES-CBC with PBKDF2 key derivation.
    Includes metadata (length prefix + json) and image png bytes.
    :param plaintext_bytes: The combined byte data (metadata_len + metadata + png_image)
    :param password: Password for encryption
    :return: Encrypted bytes (salt + iv + ciphertext) or None on error
    """
    if not PYCRYPTODOME_AVAILABLE:
        print("Error: pycryptodome is required for encryption.")
        return None
    if not password:
        print("Error: Password cannot be empty.")
        return None
    if not isinstance(plaintext_bytes, bytes):
         print("Error: Input data for encryption must be bytes.")
         return None

    try:
        # 1. Generate Salt
        salt = get_random_bytes(SALT_SIZE)

        # 2. Derive Key using PBKDF2
        key = PBKDF2(password.encode('utf-8'), salt, dkLen=KEY_SIZE, count=PBKDF2_ITERATIONS, hmac_hash_module=SHA256)

        # 3. Encrypt Data using AES-CBC
        cipher = AES.new(key, AES.MODE_CBC) # IV is generated automatically
        iv = cipher.iv

        # Pad plaintext to be multiple of block size, then encrypt
        ciphertext = cipher.encrypt(pad(plaintext_bytes, AES_BLOCK_SIZE))

        # 4. Combine Salt + IV + Ciphertext
        encrypted_data = salt + iv + ciphertext
        print(f"Encryption successful. Output size: {len(encrypted_data)} bytes (Salt: {len(salt)}, IV: {len(iv)}, Ciphertext: {len(ciphertext)})")
        return encrypted_data

    except Exception as e:
        print(f"Error during AES encryption: {e}")
        return None

# --- ComfyUI Node Class ---

class EncryptSaveAES:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output" # Mark as output type

    @classmethod
    def INPUT_TYPES(cls):
        # Ensure pycryptodome is available before defining inputs
        if not PYCRYPTODOME_AVAILABLE:
             # If library is missing, show an error message in the UI instead of inputs
            return {
                "required": {
                     "error_message": ("STRING", {
                         "multiline": True,
                         "default": "Error: pycryptodome library not found.\nPlease install it (pip install pycryptodome)\nand restart ComfyUI."
                     })
                }
            }
        return {
            "required": {
                "images": ("IMAGE", ),       # Input is image tensor
                "password": ("STRING", {"multiline": False, "default": "123"}), # Password input
                "filename_prefix": ("STRING", {"default": "ComfyUI_AES_Encrypted"}), # Filename prefix
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, # Hidden inputs for workflow data
        }

    RETURN_TYPES = () # This node doesn't pass data downstream, only saves files
    FUNCTION = "encrypt_and_save_aes"
    OUTPUT_NODE = True # Crucial: Marks this as an output node
    CATEGORY = "image/save" # Categorize under image saving

    def encrypt_and_save_aes(self, images, password, filename_prefix="ComfyUI_AES_Encrypted", prompt=None, extra_pnginfo=None, **kwargs):
        # Handle the case where pycryptodome is missing and only 'error_message' input exists
        if not PYCRYPTODOME_AVAILABLE:
             print("Execution skipped: pycryptodome is missing.")
             # Return empty UI results if the node couldn't even be properly defined
             return {"ui": {"images": []}}
        if 'error_message' in kwargs: # Check if the error message was passed due to missing lib
             print("Execution skipped: pycryptodome is missing (detected via error_message input).")
             return {"ui": {"images": []}}


        # --- Basic Input Validation ---
        if not password:
            print("Error: Password cannot be empty. Skipping save.")
            # Return empty results to UI to avoid errors downstream if any were expected
            return {"ui": {"images": []}}

        # Get the full path and filename counter etc.
        full_output_folder, filename, counter, subfolder, filename_prefix_resolved = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        for i, image in enumerate(images):
            start_time_image = time.time()
            print(f"Processing image {i+1}/{len(images)}...")

            # 1. Convert Tensor to PIL Image
            np_image = image.cpu().numpy()
            if np.max(np_image) > 1.001 or np.min(np_image) < -0.001: # Allow for small float inaccuracies
                print(f"Warning: Image {i} tensor values outside expected [0, 1] range ({np.min(np_image):.2f} - {np.max(np_image):.2f}). Clamping.")
                np_image = np.clip(np_image, 0.0, 1.0)

            img_uint8 = (np_image * 255.0).astype(np.uint8)

            pil_image = None
            try:
                if img_uint8.ndim == 3 and img_uint8.shape[2] == 4:
                    pil_image = Image.fromarray(img_uint8, 'RGBA')
                elif img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
                    pil_image = Image.fromarray(img_uint8, 'RGB')
                elif img_uint8.ndim == 2 or (img_uint8.ndim == 3 and img_uint8.shape[2] == 1):
                    if img_uint8.ndim == 3:
                        img_uint8 = np.squeeze(img_uint8, axis=2)
                    pil_image = Image.fromarray(img_uint8, 'L')
                else:
                    print(f"Error: Unsupported image shape {img_uint8.shape} for image {i}. Skipping.")
                    continue
            except Exception as e:
                 print(f"Error converting tensor to PIL Image for image {i}: {e}")
                 continue

            # 2. Save PIL Image to PNG Bytes in Memory
            try:
                buffer = io.BytesIO()
                # Use minimal compression for speed, as it won't help much after encryption anyway
                pil_image.save(buffer, format="PNG", optimize=True, compress_level=6)
                png_image_bytes = buffer.getvalue()
                buffer.close()
                print(f"Image {i} converted to PNG bytes (Size: {len(png_image_bytes) / 1024:.2f} KB)")
            except Exception as e:
                 print(f"Error saving PIL Image to PNG bytes for image {i}: {e}")
                 continue

            # 3. Prepare Metadata Bytes
            metadata_dict = {}
            if prompt is not None:
                metadata_dict["prompt"] = prompt
            if extra_pnginfo is not None:
                # Ensure extra_pnginfo items are serializable (convert non-strings if needed)
                serializable_extra = {}
                for k, v in extra_pnginfo.items():
                    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                         serializable_extra[k] = v
                    else:
                         try:
                            serializable_extra[k] = str(v) # Fallback to string conversion
                            print(f"Warning: Converted non-serializable extra_pnginfo item '{k}' to string.")
                         except Exception:
                             print(f"Warning: Skipping non-serializable extra_pnginfo item '{k}' of type {type(v)}.")
                metadata_dict["extra_pnginfo"] = serializable_extra
            
            try:
                metadata_json = json.dumps(metadata_dict, ensure_ascii=False) # Use ensure_ascii=False for broader char support
                metadata_bytes = metadata_json.encode('utf-8')
                metadata_length_bytes = len(metadata_bytes).to_bytes(4, 'big') # 4 bytes for length, big-endian
                print(f"Metadata prepared (Length: {len(metadata_bytes)} bytes)")
            except Exception as e:
                 print(f"Error serializing metadata for image {i}: {e}. Proceeding without metadata.")
                 metadata_bytes = b'' # Empty bytes if serialization fails
                 metadata_length_bytes = (0).to_bytes(4, 'big')

            # 4. Combine Plaintext Data (Length + Metadata + Image)
            plaintext_bytes = metadata_length_bytes + metadata_bytes + png_image_bytes

            # 5. Encrypt the Combined Data
            print(f"Starting AES encryption for image {i} (Plaintext size: {len(plaintext_bytes) / 1024:.2f} KB)...")
            encryption_start_time = time.time()
            encrypted_data = encrypt_data(plaintext_bytes, password)
            encryption_end_time = time.time()

            if encrypted_data is None:
                print(f"Encryption failed for image {i}. Skipping save.")
                continue # Skip saving this image if encryption failed
            print(f"Encryption for image {i} finished in {encryption_end_time - encryption_start_time:.2f} seconds.")

            # 6. Construct Output Filename (forcing .png extension)
            file = f"{filename}_{counter:05}_.png" # Use ComfyUI's counter and format, force PNG ext
            output_path = os.path.join(full_output_folder, file)

            # 7. Save Encrypted Bytes to File
            try:
                with open(output_path, 'wb') as f_out:
                    f_out.write(encrypted_data)
                print(f"Encrypted data saved to: {output_path}")
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
            except Exception as e:
                print(f"Error saving encrypted file {file}: {e}")
                # Don't increment counter if save fails, maybe? Or do? Let's increment.

            counter += 1 # Manually increment the counter for the next image
            end_time_image = time.time()
            print(f"Image {i+1} processed in {end_time_image - start_time_image:.2f} seconds.\n---")


        # Return results to the UI
        return { "ui": { "images": results } }

# --- Register Node with ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "EncryptSaveAES": EncryptSaveAES # Map class name
}

# Optional: Provide a friendlier display name
NODE_DISPLAY_NAME_MAPPINGS = {
    "EncryptSaveAES": "Encrypt and Save (AES)" # Display name in ComfyUI
}