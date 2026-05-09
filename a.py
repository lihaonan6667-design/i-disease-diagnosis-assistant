import os
from app import app, db, Case  # 根据你的项目结构导入

with app.app_context():
    # 获取所有有图片的病例
    cases = Case.query.filter(Case.image_path != None).all()
    
    upload_folder = app.config['UPLOAD_FOLDER']
    print(f"上传目录: {upload_folder}")
    
    for case in cases:
        # 拼接完整路径
        full_path = os.path.join(upload_folder, case.image_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            print(f"❌ 文件丢失: {case.image_path} (ID: {case.id})")
        else:
            print(f"✅ 文件存在: {case.image_path}")
with app.app_context():
    cases = Case.query.filter(Case.image_path != None).all()
    upload_folder = app.config['UPLOAD_FOLDER']
    
    deleted_count = 0
    for case in cases:
        full_path = os.path.join(upload_folder, case.image_path)
        if not os.path.exists(full_path):
            db.session.delete(case)
            deleted_count += 1
            
    db.session.commit()
    print(f"已清理 {deleted_count} 条无效数据")