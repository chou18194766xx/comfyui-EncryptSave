# 从你的节点文件中导入类和映射
from .encrypt_save_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 将导入的映射导出，供 ComfyUI 加载
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
