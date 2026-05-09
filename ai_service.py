import requests
import base64
import os
from config import Config

def encode_image(image_path):
    """将图片编码为base64"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            if len(image_data) == 0:
                raise ValueError(f"图片文件为空: {image_path}")
            
            # 获取图片扩展名以确定 MIME 类型
            ext = os.path.splitext(image_path)[1].lower()
            if ext == '.png':
                mime = 'image/png'
            elif ext in ['.jpg', '.jpeg']:
                mime = 'image/jpeg'
            elif ext == '.gif':
                mime = 'image/gif'
            else:
                mime = 'image/jpeg' # 默认
            
            base64_str = base64.b64encode(image_data).decode('utf-8')
            # 返回包含 MIME 类型的 Data URL
            return f"data:{mime};base64,{base64_str}"
            
    except Exception as e:
        print(f"图片编码失败: {str(e)}")
        return None # 编码失败返回 None

# ================= 新增：纯文本分析函数 =================
def analyze_text_only(description):
    """
    专门用于纯文本分析的函数
    根据配置自动选择 Qwen 或 OpenAI 格式
    """
    api_url = Config.AI_API_URL.lower()
    
    # 1. 确定请求的 URL 和 Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.AI_API_KEY}"
    }
    
    # 如果是阿里云 DashScope，使用兼容模式端点
    if 'dashscope' in api_url or 'aliyun' in api_url:
        url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        model = Config.AI_MODEL or "qwen-turbo" # 纯文本推荐用 turbo
    else:
        url = Config.AI_API_URL
        model = Config.AI_MODEL or "gpt-3.5-turbo"
    
    # 2. 构建纯文本 Prompt
    # 请根据你的实际需求修改 System Prompt
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system", 
                "content": "你是一位专业的医疗咨询助手。请根据用户的文字描述提供初步的分析和建议。"
            },
            {
                "role": "user",
                "content": f"患者描述：{description}\n\n请提供：1. 可能的原因分析 2. 建议的科室 3. 注意事项。"
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        print(f"纯文本模式：发送请求到 {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # 兼容不同 API 的返回格式
            if 'choices' in result and len(result['choices']) > 0:
                analysis_text = result['choices'][0]['message']['content']
            elif 'output' in result:
                analysis_text = result['output']
            else:
                analysis_text = str(result)
                
            return {
                'analysis': analysis_text,
                'diagnosis': analysis_text
            }
        else:
            error_msg = response.text if hasattr(response, 'text') else f'状态码: {response.status_code}'
            print(f"纯文本API调用失败: {error_msg}")
            return {
                'analysis': f'纯文本分析失败: {error_msg}',
                'diagnosis': '无法获取诊断结果'
            }
            
    except Exception as e:
        print(f"纯文本分析异常: {str(e)}")
        return {
            'analysis': f'服务调用异常: {str(e)}',
            'diagnosis': '请稍后重试'
        }

# ================= 修改：主分析函数 (核心逻辑) =================
def analyze_with_ai(image_path, description):
    """
    调用大模型API进行疾病诊断分析
    支持智能分流：
    1. 有图片 -> 多模态分析 (Qwen-VL / GPT-4V)
    2. 无图片 -> 纯文本分析 (Qwen-Turbo / GPT-3.5)
    """
    try:
        # --- 智能分流逻辑开始 ---
        
        # 判断是否有有效图片路径
        has_valid_image = image_path and os.path.exists(image_path)
        
        if has_valid_image:
            # --- 分支 A：多模态分析 (图片+文本) ---
            print(f"【多模态模式】检测到图片: {image_path}，描述: {description}")
            
            base64_image = encode_image(image_path)
            if not base64_image:
                # 图片读取失败，降级为纯文本分析
                print("警告：图片读取失败，降级为纯文本分析")
                return analyze_text_only(description)
            
            # 判断 API 类型并调用对应的多模态函数
            api_url = Config.AI_API_URL.lower()
            
            if 'dashscope' in api_url or 'qwen' in api_url or 'aliyun' in api_url:
                print("使用阿里云 Qwen-VL 进行分析")
                return analyze_with_qwen(image_path, description, base64_image)
            else:
                print("使用 OpenAI 格式 API 进行分析")
                return analyze_with_openai_format(description, base64_image)
                
        else:
            # --- 分支 B：纯文本分析 ---
            # 只有文字描述，没有图片
            print(f"【纯文本模式】无图片，仅分析描述: {description}")
            return analyze_text_only(description)

    except Exception as e:
        # 兜底异常处理，防止前端一直转圈
        error_msg = str(e)
        print(f"AI分析总异常: {error_msg}")
        return {
            'analysis': f'系统内部错误: {error_msg}',
            'diagnosis': '请稍后重试或联系管理员'
        }

# ================= 原有函数保持不变 (多模态部分) =================
def analyze_with_openai_format(description, base64_image):
    """使用OpenAI兼容格式的API调用 (多模态)"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.AI_API_KEY}"
    }

    payload = {
        "model": Config.AI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"请作为专业医生分析以下病情。\n\n**患者描述：** {description}\n\n**请根据图片和描述，提供简洁的医学分析和诊断建议。**"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # 假设 base64_image 已经包含 data:image/...;base64, 前缀
                            "url": base64_image 
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }

    try:
        response = requests.post(Config.AI_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                analysis_text = result['choices'][0]['message']['content']
            else:
                analysis_text = str(result)
            return {
                'analysis': analysis_text,
                'diagnosis': analysis_text
            }
        else:
            error_msg = response.text if hasattr(response, 'text') else f'状态码: {response.status_code}'
            return {
                'analysis': f'API调用失败: {error_msg}',
                'diagnosis': '无法获取诊断结果'
            }
    except Exception as e:
        return {
            'analysis': f'请求异常: {str(e)}',
            'diagnosis': '网络请求失败'
        }

def analyze_with_qwen(image_path, description, base64_image):
    """使用Qwen（阿里云DashScope）格式的API调用 (多模态)"""
    # 阿里云DashScope兼容OpenAI格式的API端点
    if Config.AI_API_URL and 'dashscope' in Config.AI_API_URL.lower():
        api_url = Config.AI_API_URL
    else:
        api_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.AI_API_KEY}"
    }

    # 注意：这里传入的 base64_image 应该是 encode_image 返回的完整 Data URL
    # 如果不是完整 Data URL，需要在这里拼接
    # 由于 encode_image 现在返回完整 Data URL，这里直接使用
    payload = {
        "model": Config.AI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"请作为专业医生分析以下病情。\n\n**患者描述：** {description}\n\n**请根据图片和描述，提供简洁的医学分析和诊断建议。**"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }

    try:
        print(f"发送API请求到: {api_url}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                analysis_text = result['choices'][0]['message']['content']
            elif 'output' in result:
                analysis_text = result['output']
            else:
                analysis_text = str(result)

            return {
                'analysis': analysis_text,
                'diagnosis': analysis_text
            }
        else:
            error_detail = response.text if hasattr(response, 'text') else f'状态码: {response.status_code}'
            print(f"API调用失败: {response.status_code}, 详情: {error_detail}")
            return {
                'analysis': f'API调用失败: {error_detail}',
                'diagnosis': '无法获取诊断结果'
            }
    except Exception as e:
        return {
            'analysis': f'API调用异常: {str(e)}',
            'diagnosis': '请检查网络连接'
        }