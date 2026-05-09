from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from config import Config
from models import db, User, Case
from ai_service import analyze_with_ai
import jwt
from datetime import datetime, timedelta

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# 初始化数据库
db.init_app(app)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_token(user_id):
    """生成JWT token"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """验证JWT token"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except:
        return None

def require_auth(f):
    """装饰器：要求认证"""
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '未授权访问'}), 401
        if token.startswith('Bearer '):
            token = token[7:]
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'error': '无效的token'}), 401
        request.user_id = user_id
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def require_admin(f):
    """装饰器：要求管理员权限"""
    @require_auth
    def decorated_function(*args, **kwargs):
        user = User.query.get(request.user_id)
        if not user or not user.is_admin:
            return jsonify({'error': '需要管理员权限'}), 403
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/api/register', methods=['POST'])
def register():
    """用户注册"""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'error': '请填写所有字段'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': '用户名已存在'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': '邮箱已被注册'}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    token = generate_token(user.id)
    return jsonify({
        'message': '注册成功',
        'token': token,
        'user': user.to_dict()
    }), 201

@app.route('/api/login', methods=['POST'])
def login():
    """用户登录"""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': '请填写用户名和密码'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'error': '用户名或密码错误'}), 401

    token = generate_token(user.id)
    return jsonify({
        'message': '登录成功',
        'token': token,
        'user': user.to_dict()
    })

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_case():
    """上传病例（支持纯文本、纯图片或组合）"""
    
    # 1. 安全地获取文件，不要直接访问 request.files['image']，否则没文件会报错
    file = request.files.get('image') 
    description = request.form.get('description', '')

    # 2. 校验逻辑：至少要有文字或图片之一
    if not file and not description:
        return jsonify({'error': '请上传图片或填写病情描述'}), 400

    image_path = None
    filename = None

    # 3. 【关键修改】只有当文件存在且合法时，才进行保存操作
    if file and file.filename != '' and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
    
    # 注意：如果 file 是 None 或者文件名无效，代码会直接跳过上面的块，
    # 此时 image_path 保持为 None，程序继续向下执行，不会报错。

    # 4. 创建病例记录
    case = Case(
        user_id=request.user_id,
        description=description,
        image_path=filename # 如果是纯文本，这里存 None
    )
    db.session.add(case)
    db.session.commit()

    # 5. 调用 AI 分析
    try:
        # 将路径和描述传给 AI 函数
        ai_result = analyze_with_ai(image_path, description)
        
        case.ai_analysis = ai_result.get('analysis', '')
        case.diagnosis_result = ai_result.get('diagnosis', '')
        db.session.commit()

        return jsonify({
            'message': '上传成功',
            'case': case.to_dict()
        }), 201

    except Exception as e:
        print(f"AI分析失败: {str(e)}")
        # 即使 AI 失败，也要返回成功，或者至少返回一个明确的错误 JSON
        # 这样前端才能收到数据，结束“加载中”的状态
        return jsonify({
            'message': '上传成功，但AI分析失败',
            'case': case.to_dict()
        }), 200 

@app.route('/api/my-cases', methods=['GET'])
@require_auth
def get_my_cases():
    """获取当前用户的病例列表"""
    cases = Case.query.filter_by(user_id=request.user_id).order_by(Case.created_at.desc()).all()
    return jsonify({
        'cases': [case.to_dict() for case in cases]
    })

@app.route('/api/cases/<int:case_id>', methods=['GET'])
@require_auth
def get_case(case_id):
    """获取单个病例详情"""
    case = Case.query.get_or_404(case_id)
    # 只能查看自己的病例，除非是管理员
    user = User.query.get(request.user_id)
    if case.user_id != request.user_id and not user.is_admin:
        return jsonify({'error': '无权访问'}), 403
    return jsonify(case.to_dict())

@app.route('/api/uploads/<filename>')
def uploaded_file(filename):
    """获取上传的图片"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============ 管理员接口 ============

@app.route('/api/admin/users', methods=['GET'])
@require_admin
def get_all_users():
    """获取所有用户列表"""
    users = User.query.all()
    return jsonify({
        'users': [user.to_dict() for user in users]
    })

@app.route('/api/admin/cases', methods=['GET'])
@require_admin
def get_all_cases():
    """获取所有病例列表"""
    cases = Case.query.order_by(Case.created_at.desc()).all()
    return jsonify({
        'cases': [case.to_dict() for case in cases]
    })

@app.route('/api/admin/cases/<int:case_id>', methods=['PUT'])
@require_admin
def update_case(case_id):
    """更新病例信息"""
    case = Case.query.get_or_404(case_id)
    data = request.json

    if 'description' in data:
        case.description = data['description']
    if 'diagnosis_result' in data:
        case.diagnosis_result = data['diagnosis_result']
    if 'ai_analysis' in data:
        case.ai_analysis = data['ai_analysis']

    case.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'message': '更新成功',
        'case': case.to_dict()
    })

@app.route('/api/admin/cases/<int:case_id>', methods=['DELETE'])
@require_admin
def delete_case(case_id):
    """删除病例"""
    case = Case.query.get_or_404(case_id)

    # 删除图片文件
    if case.image_path:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], case.image_path)
        if os.path.exists(filepath):
            os.remove(filepath)

    db.session.delete(case)
    db.session.commit()

    return jsonify({'message': '删除成功'})

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@require_admin
def delete_user(user_id):
    """删除用户（会级联删除其病例）"""
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': '删除成功'})

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@require_admin
def update_user(user_id):
    """更新用户信息"""
    user = User.query.get_or_404(user_id)
    data = request.json

    if 'username' in data:
        # 检查用户名是否已被其他用户使用
        existing = User.query.filter_by(username=data['username']).first()
        if existing and existing.id != user_id:
            return jsonify({'error': '用户名已存在'}), 400
        user.username = data['username']

    if 'email' in data:
        # 检查邮箱是否已被其他用户使用
        existing = User.query.filter_by(email=data['email']).first()
        if existing and existing.id != user_id:
            return jsonify({'error': '邮箱已被使用'}), 400
        user.email = data['email']

    if 'is_admin' in data:
        user.is_admin = bool(data['is_admin'])

    if 'password' in data and data['password']:
        user.set_password(data['password'])

    db.session.commit()

    return jsonify({
        'message': '更新成功',
        'user': user.to_dict()
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # 创建默认管理员账户（如果不存在）
        # 注意：生产环境请务必修改默认密码或使用 create_admin.py 创建管理员
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            import secrets
            default_password = secrets.token_urlsafe(12)  # 生成随机密码
            admin = User(username='admin', email='admin@example.com', is_admin=True)
            admin.set_password(default_password)
            db.session.add(admin)
            db.session.commit()
            print(f"默认管理员账户已创建: admin / {default_password}")
            print("⚠️  请记录此密码或使用 create_admin.py 重置管理员密码")

    app.run(debug=True, host='0.0.0.0', port=5000)

