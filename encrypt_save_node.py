import torch
import numpy as np
from PIL import Image, PngImagePlugin
import hashlib
import random
import json
import os
import folder_paths  # ComfyUI 的路径工具

# --- 从你提供的脚本中提取并调整加密逻辑 ---
# 我们将原始函数修改为接受 PIL Image 对象而不是文件路径
# 并返回加密后的 PIL Image 对象和 PngInfo 对象，而不是直接保存

def encrypt_pil_image(pil_image, password):
    """
    加密 PIL Image 对象 - 使用密码生成密钥并准备 PNG 元数据
    :param pil_image: 输入的 PIL Image 对象
    :param password: 加密密码
    :return: (encrypted_pil_image, pnginfo_object)
    """
    # 使用密码生成密钥
    key = int(hashlib.sha256(password.encode()).hexdigest(), 16) % (2**32)

    # 设置随机种子 (每次调用都重置，保证相同密码和输入图像得到相同结果)
    current_random_state = random.getstate()
    current_np_random_state = np.random.get_state()
    random.seed(key)
    np.random.seed(key)

    try:
        img_format = pil_image.format if pil_image.format else 'PNG' # 假设为PNG如果没有原始格式

        # 尝试保留原始元数据 (注意：从Tensor转换来的Image可能没有丰富元数据)
        # ComfyUI的图像张量通常不携带原始文件的所有元数据
        # 我们主要关心的是添加加密所需的信息
        metadata = pil_image.info.copy() # 获取传入 PIL Image 可能存在的元数据

        # 转换为numpy数组
        img_array = np.array(pil_image)
        if img_array.dtype != np.uint8:
             # 如果来自Tensor且范围是0-1，需要转换
             if np.max(img_array) <= 1.0 and np.min(img_array) >= 0.0:
                 img_array = (img_array * 255.0).astype(np.uint8)
             else:
                 # 尝试安全转换，或抛出错误/警告
                 img_array = img_array.astype(np.uint8)


        # 获取图像尺寸
        if img_array.ndim < 2:
            raise ValueError("Image array must have at least 2 dimensions")
        height, width = img_array.shape[:2]
        channels = img_array.shape[2] if img_array.ndim == 3 else 1

        # 块大小
        block_size = 8

        # 创建输出数组
        encrypted_array = np.zeros_like(img_array)

        # 计算块数
        num_blocks_h = height // block_size
        num_blocks_w = width // block_size

        if num_blocks_h == 0 or num_blocks_w == 0:
             # 如果图像太小，无法分块，可以选择直接处理或跳过分块加密
             # 这里我们选择直接复制，或者你可以添加简单的像素操作
             print(f"Warning: Image size ({width}x{height}) is too small for block size {block_size}. Applying simple inversion.")
             if channels >= 3: # 彩色或带Alpha
                 encrypted_array = 255 - img_array # 简单反相
             elif channels == 1: # 灰度
                 encrypted_array = 255 - img_array
             else: # 其他情况直接复制
                 encrypted_array = img_array.copy()
             block_indices = [] # 无块映射

        else:
            # 创建块索引映射
            block_indices = list(range(num_blocks_h * num_blocks_w))
            random.shuffle(block_indices)

            # 处理每个块
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    block_idx = i * num_blocks_w + j
                    target_idx = block_indices[block_idx]
                    target_i = target_idx // num_blocks_w
                    target_j = target_idx % num_blocks_w

                    i_start, i_end = i * block_size, (i + 1) * block_size
                    j_start, j_end = j * block_size, (j + 1) * block_size
                    target_i_start, target_i_end = target_i * block_size, (target_i + 1) * block_size
                    target_j_start, target_j_end = target_j * block_size, (target_j + 1) * block_size

                    block = img_array[i_start:i_end, j_start:j_end].copy()

                    # 应用颜色变换
                    if channels >= 3: # 彩色图像 (RGB 或 RGBA)
                        # 只处理RGB通道，保持Alpha不变（如果存在）
                        rgb_block = block[:, :, :3]
                        if (i + j) % 3 == 0:
                            rgb_block = rgb_block[:,:,::-1]  # 反转RGB
                        elif (i + j) % 3 == 1:
                            rgb_block[:,:,0], rgb_block[:,:,1] = rgb_block[:,:,1].copy(), rgb_block[:,:,0].copy() # Swap R<->G
                        else:
                             rgb_block[:,:,1], rgb_block[:,:,2] = rgb_block[:,:,2].copy(), rgb_block[:,:,1].copy() # Swap G<->B
                        block[:, :, :3] = rgb_block
                    elif channels == 1: # 灰度图像
                         if (i + j) % 2 == 0:
                             block = 255 - block # 反相
                    # 其他通道数（例如只有Alpha？）不处理

                    encrypted_array[target_i_start:target_i_end, target_j_start:target_j_end] = block

            # 处理图像边缘 (保持与你脚本一致的简单反转)
            h_rem = height % block_size
            w_rem = width % block_size

            if h_rem != 0:
                h_start = num_blocks_h * block_size
                encrypted_array[h_start:, :width - w_rem] = img_array[h_start:, :width - w_rem][::-1, :]
            if w_rem != 0:
                w_start = num_blocks_w * block_size
                encrypted_array[:height - h_rem, w_start:] = img_array[:height - h_rem, w_start:][:, ::-1]
            if h_rem != 0 and w_rem != 0:
                h_start = num_blocks_h * block_size
                w_start = num_blocks_w * block_size
                encrypted_array[h_start:, w_start:] = img_array[h_start:, w_start:][::-1, ::-1]

        # 转换回图片
        encrypted_img = Image.fromarray(encrypted_array, mode=pil_image.mode) # 保持原始模式 (RGB/RGBA/L)

        # 创建PNG元信息对象
        pnginfo = PngImagePlugin.PngInfo()

        # 尝试添加原始元数据中的文本信息
        for k, v in metadata.items():
            if isinstance(v, str):
                 # 避免添加可能冲突或过长的非文本信息
                 if len(v) < 1024: # 限制长度，防止意外
                    try:
                        pnginfo.add_text(k, v)
                    except Exception as e:
                        print(f"Warning: Could not add metadata key '{k}': {e}")

        # 添加块映射信息 (关键!)
        if block_indices: # 只有在成功分块时才添加
            pnginfo.add_text('block_mapping', json.dumps(block_indices))
        # 添加一个标识，表明这是加密过的文件
        pnginfo.add_text('encryption_method', 'custom_block_shuffle_v1')


    finally:
        # 恢复随机状态，避免影响 ComfyUI 其他部分
        random.setstate(current_random_state)
        np.random.set_state(current_np_random_state)

    return encrypted_img, pnginfo

# --- ComfyUI 节点类 ---

class EncryptSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output" # 标记为输出类型

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),       # 输入是图像张量
                "password": ("STRING", {"multiline": False, "default": "123"}), # 密码输入
                "filename_prefix": ("STRING", {"default": "ComfyUI_Encrypted"}), # 文件名前缀
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, # 隐藏输入，用于获取工作流信息
        }

    RETURN_TYPES = () # 这个节点不向下传递数据，只保存文件
    FUNCTION = "encrypt_and_save"
    OUTPUT_NODE = True # 关键：标记这是一个输出节点
    CATEGORY = "image/save" # 归类到图像保存类别下

    def encrypt_and_save(self, images, password, filename_prefix="ComfyUI_Encrypted", prompt=None, extra_pnginfo=None):
        # 获取完整的路径和文件名计数器等
        full_output_folder, filename, counter, subfolder, filename_prefix_resolved = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        for i, image in enumerate(images):
            # 1. 将 Tensor 转换为 NumPy array (CHW -> HWC if needed, scale 0-1 -> 0-255)
            # ComfyUI 的 image tensor 格式是 B, H, W, C, float32, range [0, 1]
            np_image = image.cpu().numpy()
            if np.max(np_image) > 1.0 or np.min(np_image) < 0.0:
                print("Warning: Input image tensor values are outside the expected [0, 1] range. Clamping.")
                np_image = np.clip(np_image, 0.0, 1.0)

            img_uint8 = (np_image * 255.0).astype(np.uint8)

            # 2. 将 NumPy array 转换为 PIL Image
            # 检测模式 (RGB vs RGBA)
            if img_uint8.shape[2] == 4:
                pil_image = Image.fromarray(img_uint8, 'RGBA')
            elif img_uint8.shape[2] == 3:
                pil_image = Image.fromarray(img_uint8, 'RGB')
            elif img_uint8.shape[2] == 1:
                 # 处理单通道 (灰度)
                 # 如果维度是 (H, W, 1)，需要去掉最后一个维度
                 if img_uint8.ndim == 3 and img_uint8.shape[2] == 1:
                      img_uint8 = np.squeeze(img_uint8, axis=2)
                 pil_image = Image.fromarray(img_uint8, 'L')
            else:
                print(f"Error: Unsupported number of channels ({img_uint8.shape[2]}) for image {i}. Skipping.")
                continue # 跳过这个不支持的图像


            # 3. 调用加密函数
            if not password:
                print("Error: Password cannot be empty. Skipping encryption for image {i}.")
                # 可以选择是跳过、保存未加密还是抛出错误
                # 这里我们跳过保存
                continue

            encrypted_pil_image, pnginfo_obj = encrypt_pil_image(pil_image, password)

            # 添加来自工作流的元数据 (prompt 和 extra_pnginfo)
            if prompt is not None:
                pnginfo_obj.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    # 确保只添加文本信息
                    if isinstance(v, str) and len(v) < 1024 : # 再次检查，避免非字符串或过长内容
                       try:
                            pnginfo_obj.add_text(k, v)
                       except Exception as e:
                           print(f"Warning: Could not add extra_pnginfo key '{k}': {e}")

            # 4. 构建输出文件名
            file = f"{filename}_{counter:05}_.png"
            output_path = os.path.join(full_output_folder, file)

            # 5. 保存加密后的图像和元数据
            try:
                encrypted_pil_image.save(output_path, pnginfo=pnginfo_obj, optimize=False, compress_level=4) # 使用适中的压缩级别
                print(f"Encrypted image saved to: {output_path}")
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
            except Exception as e:
                print(f"Error saving encrypted image {file}: {e}")

            counter += 1 # 手动增加计数器，因为我们是在循环内部处理

        # 返回结果给 UI，以便显示保存的文件
        return { "ui": { "images": results } }

# --- 注册节点到 ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "EncryptSaveImage": EncryptSaveImage # 映射类名
}

# 可选：提供一个更友好的显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "EncryptSaveImage": "Encrypt and Save Image" # 节点在界面上的名字
}