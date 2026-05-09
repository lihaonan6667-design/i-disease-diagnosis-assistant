import os
from pathlib import Path

from dotenv import load_dotenv

# 始终从本文件所在目录加载 .env（避免在别的 cwd 下启动时读不到密钥）
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///disease_diagnosis.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # 相对路径会落到「启动 Python 时的 cwd」，易导致文件不在项目 uploads 下
    UPLOAD_FOLDER = str(_BASE_DIR / 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

    # 大模型API配置（可通过环境变量修改）
    # 阿里云通义千问配置
    # 注意：生产环境请务必通过环境变量设置AI_API_KEY，不要硬编码密钥
    # 如果未设置，将使用占位符（实际使用时需要在.env文件中配置）
    AI_API_KEY = (os.environ.get('AI_API_KEY') or '').strip() or 'your-api-key-here-please-set-in-env'
    # 阿里云DashScope兼容OpenAI格式的API端点
    AI_API_URL = (os.environ.get('AI_API_URL') or '').strip() or 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
    # 通义千问多模态模型（支持图片分析）
    # qwen-vl-plus: 视觉模型，支持图片输入（推荐）
    # qwen-vl-max: 多模态模型，支持图片输入
    # qwen-max: 纯文本模型，不支持图片
    AI_MODEL = (os.environ.get('AI_MODEL') or '').strip() or 'qwen3.5-omni-flash'

