# 疾病诊断系统

**英文题目**：Design and Implementation of an AI Large Model-based Disease Diagnosis Assistance System  

**建议 GitHub 仓库名（短）**：`ai-disease-diagnosis-assistant`

一个基于Web的智能疾病诊断辅助系统，支持用户上传患病照片和病情描述，通过大模型API进行智能分析。

## 功能特性

- ✅ 用户注册和登录
- ✅ 上传患病照片和病情描述
- ✅ AI智能诊断分析
- ✅ 查看个人病例历史记录
- ✅ 管理员后台管理（用户管理、病例管理）

## 技术栈

- **前端**：HTML5 + CSS3 + JavaScript，Bootstrap 5.3.0
- **后端**：Python 3.x + Flask 3.0.0
- **数据库**：SQLite
- **AI服务**：支持多种云服务大模型API（默认阿里云DashScope）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `config_example.txt` 为 `.env` 并修改配置：

```bash
# Windows
copy config_example.txt .env

# Linux/Mac
cp config_example.txt .env
```

编辑 `.env` 文件，设置你的API密钥：

```env
AI_API_KEY=your-api-key-here
AI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
AI_MODEL=qwen-vl-plus
```

### 3. 创建管理员账户

```bash
python create_admin.py
```

脚本会生成一个随机密码，请妥善保存。

### 4. 启动后端服务

```bash
python app.py
```

后端服务将在 `http://localhost:5000` 启动。

### 5. 打开前端页面

在浏览器中打开 `index.html` 文件，或使用本地服务器：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000`

## 项目结构

```
.
├── app.py              # Flask后端主文件
├── models.py           # 数据库模型
├── config.py           # 配置文件
├── ai_service.py       # AI服务调用
├── create_admin.py     # 管理员账户管理工具
├── requirements.txt    # Python依赖
├── config_example.txt  # 环境变量配置示例
├── index.html          # 登录/注册页面
├── user.html           # 用户主界面
├── admin.html          # 管理员界面
├── uploads/            # 上传图片存储目录（自动创建）
└── instance/           # 数据库文件目录（自动创建）
```