import torch
import numpy as np
from PIL import Image, PngImagePlugin
import hashlib
import random
import json
import os
import time # 引入 time 模块，虽然不在核心逻辑中使用，但保留可能性
import folder_paths  # ComfyUI 的路径工具

# --- 采用参考脚本的加密逻辑并适配为函数 ---

def encrypt_pil_image_optimized(pil_image, password, block_size=64):
    """
    加密 PIL Image 对象 - 使用参考脚本的优化逻辑
    :param pil_image: 输入的 PIL Image 对象 (应为 uint8 类型)
    :param password: 加密密码
    :param block_size: 块大小
    :return: (encrypted_pil_image, pnginfo_object)
    """
    start_time = time.time() # 用于调试或记录耗时

    # --- 1. 密钥生成和随机种子设置 ---
    key = int(hashlib.sha256(password.encode()).hexdigest(), 16) % (2**32)
    # 保存当前随机状态
    current_random_state = random.getstate()
    current_np_random_state = np.random.get_state()
    random.seed(key)
    np.random.seed(key)

    try:
        # --- 2. 图像信息获取与准备 ---
        img_format = pil_image.format if pil_image.format else 'PNG' # 保留格式信息，主要用于保存
        width, height = pil_image.size
        print(f"Encrypting image with size: {width}x{height} pixels, block size: {block_size}")

        # 保留传入 PIL Image 可能存在的元数据
        metadata = pil_image.info.copy()

        # --- 3. 转换为 NumPy 数组 (应确保输入是 uint8) ---
        # 使用 np.asarray 以可能避免复制
        img_array = np.asarray(pil_image)

        # 确认数据类型，如果不是 uint8 则发出警告或尝试转换（理论上在调用此函数前应已完成）
        if img_array.dtype != np.uint8:
             print(f"Warning: encrypt_pil_image_optimized received non-uint8 NumPy array (dtype: {img_array.dtype}). Performance might be affected or errors could occur.")
             # 尝试强制转换，但这可能不是最佳位置
             img_array = img_array.astype(np.uint8)

        # --- 4. 计算块和有效区域 ---
        valid_height = (height // block_size) * block_size
        valid_width = (width // block_size) * block_size
        num_blocks_h = valid_height // block_size
        num_blocks_w = valid_width // block_size
        total_blocks = num_blocks_h * num_blocks_w

        if total_blocks == 0:
            print(f"Warning: Image size ({width}x{height}) is too small for block size {block_size}. Applying simple inversion.")
            encrypted_array = 255 - img_array # 简单反相处理小图像
            block_indices = [] # 没有块映射
        else:
            print(f"Total blocks to shuffle: {total_blocks} (Rows: {num_blocks_h}, Cols: {num_blocks_w})")
            # --- 5. 预分配内存和块索引 ---
            encrypted_array = np.empty_like(img_array) # 预分配输出数组
            block_indices = list(range(total_blocks))
            random.shuffle(block_indices)

            # --- 6. 加密核心过程：块置换和颜色变换 ---
            shuffle_start = time.time()
            print("Starting block processing...")

            img_ndim = img_array.ndim
            img_channels = img_array.shape[2] if img_ndim == 3 else 1

            for block_idx in range(total_blocks):
                # 计算原始块位置
                i = block_idx // num_blocks_w
                j = block_idx % num_blocks_w
                i_start, i_end = i * block_size, (i + 1) * block_size
                j_start, j_end = j * block_size, (j + 1) * block_size

                # 计算目标块位置
                target_idx = block_indices[block_idx]
                target_i = target_idx // num_blocks_w
                target_j = target_idx % num_blocks_w
                target_i_start, target_i_end = target_i * block_size, (target_i + 1) * block_size
                target_j_start, target_j_end = target_j * block_size, (target_j + 1) * block_size

                # 获取块视图 (避免立即复制)
                block = img_array[i_start:i_end, j_start:j_end]

                # 应用颜色变换 (只在必要时复制)
                transformed_block = None
                if img_ndim == 3 and img_channels >= 3:  # 彩色图像 (RGB 或 RGBA)
                    # 只处理 RGB 通道
                    if (i + j) % 3 == 0:
                        # 反转 RGB (注意：如果RGBA，Alpha通道不变)
                        transformed_block = block.copy()
                        transformed_block[:,:,:3] = transformed_block[:,:,:3][:,:,::-1]
                    elif (i + j) % 3 == 1:
                        # Swap R <-> G
                        transformed_block = block.copy()
                        transformed_block[:,:,0], transformed_block[:,:,1] = block[:,:,1].copy(), block[:,:,0].copy() # 使用 .copy() 确保交换
                    else:
                        # Swap G <-> B
                        transformed_block = block.copy()
                        transformed_block[:,:,1], transformed_block[:,:,2] = block[:,:,2].copy(), block[:,:,1].copy() # 使用 .copy() 确保交换
                elif img_ndim == 2:  # 灰度图像
                    if (i + j) % 2 == 0:
                        transformed_block = 255 - block # 反相
                    # else: # 不需要变换，保持原样
                # 其他情况 (例如只有Alpha通道？) 或不需要变换时，transformed_block 保持 None

                # 放置块到加密数组
                if transformed_block is not None:
                    encrypted_array[target_i_start:target_i_end, target_j_start:target_j_end] = transformed_block
                else:
                    encrypted_array[target_i_start:target_i_end, target_j_start:target_j_end] = block #直接放置原始块

            shuffle_end = time.time()
            print(f"Block processing took: {shuffle_end - shuffle_start:.2f} seconds")

            # --- 7. 处理图像边缘 ---
            # (采用参考代码的逻辑)
            if height > valid_height: # 底部边缘
                encrypted_array[valid_height:, :valid_width] = img_array[valid_height:, :valid_width][::-1, :] # 上下翻转
            if width > valid_width: # 右侧边缘
                encrypted_array[:valid_height, valid_width:] = img_array[:valid_height, valid_width:][:, ::-1] # 左右翻转
            if height > valid_height and width > valid_width: # 右下角
                encrypted_array[valid_height:, valid_width:] = img_array[valid_height:, valid_width:][::-1, ::-1] # 中心翻转

        # --- 8. 转换回 PIL Image ---
        encrypted_img = Image.fromarray(encrypted_array, mode=pil_image.mode) # 保持原始模式

        # --- 9. 创建 PNG 元信息 ---
        pnginfo = PngImagePlugin.PngInfo()

        # 尝试添加原始元数据中的文本信息
        for k, v in metadata.items():
            # 仅添加字符串类型的元数据，并做长度限制
            if isinstance(v, str) and len(v) < 1024:
                try:
                    # 避免添加可能冲突的内置键 (如 'icc_profile') - PngInfo会自动处理一些
                    if k not in ['icc_profile', 'interlace', 'gamma', 'srgb', 'aspect', 'Software']: # 可以根据需要添加更多排除项
                         pnginfo.add_text(k, v)
                except Exception as e:
                    print(f"Warning: Could not add metadata key '{k}' from original image: {e}")

        # 添加关键的加密元数据
        if total_blocks > 0: # 只有在成功分块时才添加映射
             pnginfo.add_text('block_mapping', json.dumps(block_indices))
        pnginfo.add_text('block_size', str(block_size))
        pnginfo.add_text('encryption_method', 'custom_block_shuffle_v2') # 更新版本标识

    finally:
        # --- 10. 恢复随机状态 ---
        random.setstate(current_random_state)
        np.random.set_state(current_np_random_state)

    end_time = time.time()
    print(f"Encryption function finished in {end_time - start_time:.2f} seconds")
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
                "block_size": ("INT", {"default": 8, "min": 8, "max": 256, "step": 8}), # 新增：块大小
                "filename_prefix": ("STRING", {"default": "ComfyUI_Encrypted"}), # 文件名前缀
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, # 隐藏输入，用于获取工作流信息
        }

    RETURN_TYPES = () # 这个节点不向下传递数据，只保存文件
    FUNCTION = "encrypt_and_save"
    OUTPUT_NODE = True # 关键：标记这是一个输出节点
    CATEGORY = "image/save" # 归类到图像保存类别下

    def encrypt_and_save(self, images, password, block_size, filename_prefix="ComfyUI_Encrypted", prompt=None, extra_pnginfo=None):
        # 获取完整的路径和文件名计数器等
        full_output_folder, filename, counter, subfolder, filename_prefix_resolved = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        for i, image in enumerate(images):
            # 1. Tensor 转换为 NumPy array (CHW -> HWC if needed, scale 0-1 -> 0-255)
            # ComfyUI 的 image tensor 格式是 B, H, W, C, float32, range [0, 1]
            np_image = image.cpu().numpy()
            if np.max(np_image) > 1.0 or np.min(np_image) < 0.0:
                print("Warning: Input image tensor values are outside the expected [0, 1] range. Clamping.")
                np_image = np.clip(np_image, 0.0, 1.0)

            # 转换到 uint8
            img_uint8 = (np_image * 255.0).astype(np.uint8)

            # 2. NumPy array 转换为 PIL Image (确定模式)
            pil_image = None
            if img_uint8.ndim == 3 and img_uint8.shape[2] == 4:
                pil_image = Image.fromarray(img_uint8, 'RGBA')
            elif img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
                pil_image = Image.fromarray(img_uint8, 'RGB')
            elif img_uint8.ndim == 2 or (img_uint8.ndim == 3 and img_uint8.shape[2] == 1):
                 # 处理单通道 (灰度)
                 if img_uint8.ndim == 3:
                      img_uint8 = np.squeeze(img_uint8, axis=2)
                 pil_image = Image.fromarray(img_uint8, 'L')
            else:
                print(f"Error: Unsupported image shape or channel count ({img_uint8.shape}) for image {i}. Skipping.")
                continue # 跳过这个不支持的图像

            # 3. 检查密码
            if not password:
                print(f"Error: Password cannot be empty. Skipping encryption for image {i}.")
                # 可以选择保存未加密或跳过，这里选择跳过
                continue

            # 4. 调用新的优化加密函数
            try:
                encrypted_pil_image, pnginfo_obj = encrypt_pil_image_optimized(pil_image, password, block_size)
            except Exception as e:
                print(f"Error during encryption for image {i}: {e}")
                continue # 加密失败则跳过

            # 5. 添加来自工作流的元数据 (prompt 和 extra_pnginfo)
            if prompt is not None:
                try:
                    pnginfo_obj.add_text("prompt", json.dumps(prompt))
                except Exception as e:
                     print(f"Warning: Could not add prompt metadata: {e}")
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    # 确保只添加文本信息
                    if isinstance(v, (str, int, float)) and k not in ['block_mapping', 'block_size', 'encryption_method']: # 避免覆盖加密信息
                       try:
                            # 对非字符串尝试转换，并限制长度
                            v_str = str(v)
                            if len(v_str) < 2048: # 增加一点长度限制
                                pnginfo_obj.add_text(k, v_str)
                            else:
                                print(f"Warning: Skipping long extra_pnginfo key '{k}'")
                       except Exception as e:
                           print(f"Warning: Could not add extra_pnginfo key '{k}': {e}")

            # 6. 构建输出文件名
            file = f"{filename}_{counter:05}_.png" # 强制保存为 PNG
            output_path = os.path.join(full_output_folder, file)

            # 7. 保存加密后的图像和元数据 (使用参考代码的优化参数)
            try:
                encrypted_pil_image.save(
                    output_path,
                    pnginfo=pnginfo_obj,
                    optimize=False,      # 根据参考代码的建议
                    compress_level=1     # 使用较低的压缩级别以提高速度
                )
                print(f"Encrypted image saved to: {output_path}")
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
            except Exception as e:
                print(f"Error saving encrypted image {file}: {e}")

            counter += 1 # 手动增加计数器

        # 返回结果给 UI
        return { "ui": { "images": results } }

# --- 注册节点到 ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "EncryptSaveImage": EncryptSaveImage # 映射类名
}

# 可选：提供一个更友好的显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "EncryptSaveImage": "Encrypt and Save Image (Optimized)" # 更新显示名称以反映变化
}
