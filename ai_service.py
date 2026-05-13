import requests
import base64
import json
import os
import re
from config import Config
from rag_service import rag_prefix_for_prompt, retrieve_for_query

# 预设医学专科体系（阶段一零样本分类可选类别）
MEDICAL_CATEGORIES = ["心血管疾病", "呼吸系统疾病", "消化系统疾病", "其他"]


def encode_image(image_path):
    """将图片编码为base64"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            if len(image_data) == 0:
                raise ValueError(f"图片文件为空: {image_path}")

            ext = os.path.splitext(image_path)[1].lower()
            if ext == '.png':
                mime = 'image/png'
            elif ext in ['.jpg', '.jpeg']:
                mime = 'image/jpeg'
            elif ext == '.gif':
                mime = 'image/gif'
            elif ext == '.webp':
                mime = 'image/webp'
            else:
                mime = 'image/jpeg'

            base64_str = base64.b64encode(image_data).decode('utf-8')
            return f"data:{mime};base64,{base64_str}"

    except Exception as e:
        print(f"图片编码失败: {str(e)}")
        return None


def _chat_url_and_model():
    api_url = (Config.AI_API_URL or '').lower()
    if 'dashscope' in api_url or 'aliyun' in api_url:
        return 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', Config.AI_MODEL
    return Config.AI_API_URL, Config.AI_MODEL


def _chat_completion(messages, max_tokens=512, temperature=0.3):
    """通用文本 Chat 调用，返回 assistant 文本或 None"""
    url, model = _chat_url_and_model()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.AI_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        if response.status_code != 200:
            print(f"_chat_completion HTTP {response.status_code}: {response.text[:400]}")
            return None
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            return (data['choices'][0].get('message') or {}).get('content') or ''
    except Exception as e:
        print(f"_chat_completion 异常: {e}")
    return None


def _parse_json_object(text):
    if not text:
        return None
    raw = text.strip()
    block = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if block:
        raw = block.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def step1_classify(user_text: str) -> str:
    """阶段一：零样本专科归类"""
    cats = MEDICAL_CATEGORIES
    prompt = f"""将用户症状归类到下列**唯一**类别（必须逐字匹配列表中的一项）：
{json.dumps(cats, ensure_ascii=False)}

用户描述：{user_text}

只输出 JSON，不要其它文字：{{"category":"..."}}"""
    content = _chat_completion(
        [
            {"role": "system", "content": "你是医疗分诊助手。category 必须是用户给定列表中的确切一项。只输出合法 JSON。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
        temperature=0.2,
    )
    obj = _parse_json_object(content or '')
    if obj and isinstance(obj.get('category'), str):
        c = obj['category'].strip()
        if c in cats:
            return c
    return "其他"


def step2_keywords(user_text: str, category: str):
    """阶段二：口语 → 标准检索用词"""
    prompt = f"""初步专科：{category}
用户原话：{user_text}

请给出 3～5 个用于医学资料检索的标准中文术语或名词短语（不要完整句子）。

只输出 JSON：{{"keywords":["术语1","术语2",...]}}"""
    content = _chat_completion(
        [
            {"role": "system", "content": "只输出合法 JSON，keywords 为 3～5 个简短中文名词短语。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    obj = _parse_json_object(content or '')
    if obj and isinstance(obj.get('keywords'), list):
        kw = [str(x).strip() for x in obj['keywords'] if str(x).strip()]
        return kw[:8]
    return []


def run_triage_pipeline(description: str):
    """有病情描述时运行阶段一、二；返回 (专科, 关键词列表)"""
    d = (description or '').strip()
    if not d:
        return "", []
    cat = step1_classify(d)
    kw = step2_keywords(d, cat)
    print(f"【预检】专科={cat}, 关键词={kw}")
    return cat, kw


def _format_triage_hint(triage_category, keywords):
    lines = []
    if triage_category:
        lines.append(f"初步专科归类：{triage_category}")
    if keywords:
        lines.append(f"术语关键词：{', '.join(keywords)}")
    if not lines:
        return ""
    return "\n\n" + "\n".join(lines) + "\n"


def analyze_text_only(
    description,
    triage_category="",
    keywords=None,
    rag_context="",
    *,
    brief_mode: bool = False,
):
    """纯文本最终分析（可将阶段一、二结果写入上下文）

    brief_mode: 评估用「短答弱基线」——更短输出、不鼓励堆砌术语；仅应由消融脚本在显式开关下使用。"""
    keywords = keywords or []
    api_url = Config.AI_API_URL.lower()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.AI_API_KEY}",
    }
    if 'dashscope' in api_url or 'aliyun' in api_url:
        url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        model = Config.AI_MODEL or "qwen-turbo"
    else:
        url = Config.AI_API_URL
        model = Config.AI_MODEL or "gpt-3.5-turbo"

    hint = _format_triage_hint(triage_category, keywords)
    prefix = rag_prefix_for_prompt(rag_context)
    if brief_mode:
        user_body = prefix + (
            f"患者描述：{description}{hint}\n"
            "请用**一段话、约150～280字**简要说明：可能方向、建议就诊科室、一条注意事项。\n"
            "**不要使用编号分点长列表**，不要刻意罗列多个教材式专业病名与缩写；表达通俗即可。\n"
            "内容不可替代线下诊疗。"
        )
        max_tokens = 360
        temperature = 0.55
    else:
        user_body = prefix + (
            f"患者描述：{description}{hint}\n"
            "请结合上述信息（若有专科与术语提示请优先保持一致），提供："
            "1. 可能的原因分析 2. 建议就诊科室 3. 注意事项。\n"
            "内容需通俗、严谨，不可替代线下诊疗。"
        )
        max_tokens = 1200
        temperature = 0.7

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一位专业的医疗咨询助手。"},
            {"role": "user", "content": user_body},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        print(f"纯文本模式：发送请求到 {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=60)

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
                'diagnosis': analysis_text,
            }
        error_msg = response.text if hasattr(response, 'text') else f'状态码: {response.status_code}'
        print(f"纯文本API调用失败: {error_msg}")
        return {
            'analysis': f'纯文本分析失败: {error_msg}',
            'diagnosis': '无法获取诊断结果',
        }

    except Exception as e:
        print(f"纯文本分析异常: {str(e)}")
        return {
            'analysis': f'服务调用异常: {str(e)}',
            'diagnosis': '请稍后重试',
        }


def analyze_with_ai(image_path, description, skip_rag=False, eval_weak_no_rag=False):
    """
    主入口：有文字描述时先跑阶段一、二；再结合图文调用最终分析。
    仅图片无描述时不跑预检流水线。
    skip_rag: 为 True 时不检索（与 Config.RAG_ENABLED=0 配合做毕设消融实验）。
    eval_weak_no_rag: 仅当 skip_rag=True 且由 eval/run_ablation.py 显式开启时有效。
        无检索基线改为「仅主诉 + 短答约束」，不注入分诊/关键词提示，用于论文中与完整 RAG 流水线对比；
        不影响 App 正常调用（默认 False）。
    """
    eval_weak_no_rag = bool(eval_weak_no_rag and skip_rag)
    triage_category, keywords = "", []
    desc_stripped = (description or '').strip()
    rag_context, rag_sources = "", []

    try:
        if desc_stripped:
            triage_category, keywords = run_triage_pipeline(desc_stripped)

        query_parts = []
        if desc_stripped:
            query_parts.append(desc_stripped)
        if triage_category:
            query_parts.append(triage_category)
        if keywords:
            query_parts.extend(keywords)
        use_rag = Config.RAG_ENABLED and not skip_rag and bool(query_parts)
        if use_rag:
            try:
                rag_context, rag_sources = retrieve_for_query(" ".join(query_parts))
            except Exception as rag_ex:
                print(f"RAG 检索异常: {rag_ex}")

        has_valid_image = image_path and os.path.exists(image_path)

        if has_valid_image:
            print(f"【多模态模式】检测到图片: {image_path}，描述: {description!r}")

            base64_image = encode_image(image_path)
            if not base64_image:
                print("警告：图片读取失败，降级为纯文本分析")
                out = analyze_text_only(
                    description or '', triage_category, keywords, rag_context
                )
            else:
                api_url = Config.AI_API_URL.lower()
                if 'dashscope' in api_url or 'qwen' in api_url or 'aliyun' in api_url:
                    out = analyze_with_qwen(
                        description or '',
                        base64_image,
                        triage_category,
                        keywords,
                        rag_context,
                    )
                else:
                    out = analyze_with_openai_format(
                        description or '',
                        base64_image,
                        triage_category,
                        keywords,
                        rag_context,
                    )
        else:
            print(f"【纯文本模式】无图片，仅分析描述: {description!r}")
            if skip_rag and eval_weak_no_rag:
                out = analyze_text_only(
                    desc_stripped or "",
                    "",
                    [],
                    "",
                    brief_mode=True,
                )
            else:
                out = analyze_text_only(
                    description or "", triage_category, keywords, rag_context
                )

        out['triage_category'] = triage_category if triage_category else None
        out['keywords'] = keywords or []
        out['rag_sources'] = rag_sources
        return out

    except Exception as e:
        error_msg = str(e)
        print(f"AI分析总异常: {error_msg}")
        return {
            'analysis': f'系统内部错误: {error_msg}',
            'diagnosis': '请稍后重试或联系管理员',
            'triage_category': triage_category if triage_category else None,
            'keywords': keywords or [],
            'rag_sources': rag_sources,
        }


def analyze_with_openai_format(
    description, base64_image, triage_category='', keywords=None, rag_context=''
):
    keywords = keywords or []
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.AI_API_KEY}",
    }

    hint = _format_triage_hint(triage_category, keywords)
    prefix = rag_prefix_for_prompt(rag_context)
    text_part = prefix + (
        f"请作为专业医生分析以下病情。\n\n**患者描述：** {description or '（用户未填写文字，请结合图像综合判断）'}"
        f"{hint}\n**请根据图片与上述信息，提供简洁的医学分析与就诊建议。**"
    )

    payload = {
        "model": Config.AI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_part},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],
            }
        ],
        "max_tokens": 2000,
    }

    try:
        response = requests.post(Config.AI_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                analysis_text = result['choices'][0]['message']['content']
            else:
                analysis_text = str(result)
            return {
                'analysis': analysis_text,
                'diagnosis': analysis_text,
            }
        error_msg = response.text if hasattr(response, 'text') else f'状态码: {response.status_code}'
        return {
            'analysis': f'API调用失败: {error_msg}',
            'diagnosis': '无法获取诊断结果',
        }
    except Exception as e:
        return {
            'analysis': f'请求异常: {str(e)}',
            'diagnosis': '网络请求失败',
        }


def analyze_with_qwen(
    description, base64_image, triage_category='', keywords=None, rag_context=''
):
    keywords = keywords or []
    if Config.AI_API_URL and 'dashscope' in Config.AI_API_URL.lower():
        api_url = Config.AI_API_URL
    else:
        api_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.AI_API_KEY}",
    }

    hint = _format_triage_hint(triage_category, keywords)
    prefix = rag_prefix_for_prompt(rag_context)
    text_part = prefix + (
        f"请作为专业医生分析以下病情。\n\n**患者描述：** {description or '（用户未填写文字，请结合图像综合判断）'}"
        f"{hint}\n**请根据图片与上述信息，提供简洁的医学分析与就诊建议。**"
    )

    payload = {
        "model": Config.AI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_part},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],
            }
        ],
        "max_tokens": 2000,
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
                'diagnosis': analysis_text,
            }
        error_detail = response.text if hasattr(response, 'text') else f'状态码: {response.status_code}'
        print(f"API调用失败: {response.status_code}, 详情: {error_detail}")
        return {
            'analysis': f'API调用失败: {error_detail}',
            'diagnosis': '无法获取诊断结果',
        }
    except Exception as e:
        return {
            'analysis': f'API调用异常: {str(e)}',
            'diagnosis': '请检查网络连接',
        }
