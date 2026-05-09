import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///disease_diagnosis.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    # 大模型API配置（可通过环境变量修改）
    # 阿里云通义千问配置
    # 注意：生产环境请务必通过环境变量设置AI_API_KEY，不要硬编码密钥
    # 如果未设置，将使用占位符（实际使用时需要在.env文件中配置）
    AI_API_KEY = os.environ.get('AI_API_KEY') or 'your-api-key-here-please-set-in-env'
    # 阿里云DashScope兼容OpenAI格式的API端点
    AI_API_URL = os.environ.get('AI_API_URL') or 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
    # 通义千问多模态模型（支持图片分析）
    # qwen-vl-plus: 视觉模型，支持图片输入（推荐）
    # qwen-vl-max: 多模态模型，支持图片输入
    # qwen-max: 纯文本模型，不支持图片
    AI_MODEL = os.environ.get('AI_MODEL') or 'qwen-vl-plus'

