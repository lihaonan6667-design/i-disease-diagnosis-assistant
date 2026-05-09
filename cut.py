import pdfplumber
import jieba
import logging
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from paddleocr import PaddleOCR

# 屏蔽 PaddleOCR 和 jieba 的冗余日志输出
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)

# 适配 PaddleOCR 3.x 的初始化参数
ocr = PaddleOCR(use_textline_orientation=True, lang='ch')

def split_pdf_to_chunks_with_ocr(pdf_path, chunk_size=500, chunk_overlap=50):
    print(f"正在读取PDF文件: {pdf_path} ...")
    full_text = ""
    
    # 1. 先用 pdfplumber 尝试提取纯文本
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    
    # 2. 如果提取的文本极少，启动 OCR 兜底
    if len(full_text.strip()) < 50:
        print("⚠️ 检测到文档可能为扫描件或文本极少，正在启动 OCR 引擎进行文字识别（请耐心等待）...")
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 将 PDF 页面转为图片 (PIL Image 对象)
                pil_image = page.to_image(resolution=200).original
                
                # 将 PIL Image 转换为 numpy 数组 (PaddleOCR 3.x 依然需要此格式)
                cv_image = np.array(pil_image)
                
                # --- 适配 PaddleOCR 3.x 的核心修改 ---
                # 使用 predict 方法进行识别
                ocr_result = ocr.predict(cv_image)
                
                # 提取识别结果 (修复了之前的 AttributeError)
                for res in ocr_result:
                    # 在 PaddleOCR 3.x 中，识别出的文本列表通常在 res['rec_texts'] 中
                    if isinstance(res, dict) and 'rec_texts' in res:
                        for text in res['rec_texts']:
                            full_text += text + "\n"
        print(f"OCR 识别完成！")

    print(f"文本提取完成，总字符数: {len(full_text)}")
    if len(full_text) == 0:
        print("❌ 依然未提取到任何文本，请检查 PDF 文件是否损坏或为纯图片。")
        return []

    # 3. 使用 jieba 分词 + LangChain 切分
    print("正在使用 jieba 进行中文分词...")
    words = jieba.lcut(full_text, cut_all=False, HMM=True)
    segmented_text = " ".join(words)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""] 
    )
    
    chunks = text_splitter.split_text(segmented_text)
    print(f"切分完成！共生成 {len(chunks)} 个文本片段。")
    
    return chunks

# --- 使用示例 ---
if __name__ == "__main__":
    # 记得加上 r 防止路径转义报错，请确保该路径下存在你的 PDF 文件
    pdf_file = r"D:\桌面\1\1.pdf" 
    
    document_chunks = split_pdf_to_chunks_with_ocr(pdf_file, chunk_size=400, chunk_overlap=50)

    print("\n--- 预览前3个切分片段 ---")
    for i, chunk in enumerate(document_chunks[:3]):
        print(f"\n[片段 {i+1}] (长度: {len(chunk)} 字):")
        print(chunk)