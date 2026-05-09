"""
创建或检查管理员账户的工具脚本
运行方式: python create_admin.py
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, ensure_case_extra_columns
from models import db, User

with app.app_context():
    db.create_all()
    ensure_case_extra_columns()
    # 检查是否存在admin账户
    admin = User.query.filter_by(username='admin').first()

    if admin:
        print(f"管理员账户已存在:")
        print(f"  用户名: {admin.username}")
        print(f"  邮箱: {admin.email}")
        print(f"  是否管理员: {admin.is_admin}")
        print(f"\n如果要重置密码，请输入 'y'，否则按回车退出")
        choice = input().strip().lower()
        if choice == 'y':
            import secrets
            new_password = secrets.token_urlsafe(12)  # 生成随机密码
            admin.set_password(new_password)
            db.session.commit()
            print(f"密码已重置为: {new_password}")
            print("⚠️  请妥善保存此密码")
    else:
        print("创建默认管理员账户...")
        import secrets
        default_password = secrets.token_urlsafe(12)  # 生成随机密码
        admin = User(username='admin', email='admin@example.com', is_admin=True)
        admin.set_password(default_password)
        db.session.add(admin)
        db.session.commit()
        print("✓ 管理员账户创建成功!")
        print(f"  用户名: admin")
        print(f"  密码: {default_password}")
        print("⚠️  请妥善保存此密码")

    # 显示所有管理员账户
    print("\n所有管理员账户列表:")
    admins = User.query.filter_by(is_admin=True).all()
    if admins:
        for admin in admins:
            print(f"  - {admin.username} ({admin.email})")
    else:
        print("  暂无管理员账户")
