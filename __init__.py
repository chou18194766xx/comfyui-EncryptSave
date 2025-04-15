# __init__.py

# Import classes and mappings from the node file
from .encrypt_decrypt_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export the imported mappings for ComfyUI to load
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("--- Loading Custom Encrypt/Decrypt Nodes ---")
