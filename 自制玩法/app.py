import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import base64
from PIL import Image
import io

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

st.set_page_config(page_title="小李飞刀AI助手", page_icon="✨", layout="wide")
st.title("小李飞刀的APP")

# 自定义CSS样式
st.markdown("""
<style>
    /* 用户消息气泡 */
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* 助手消息气泡 */
    .assistant-bubble {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 70%;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(245, 87, 108, 0.3);
    }
    
    /* 用户消息容器 */
    .user-message {
        display: flex;
        justify-content: flex-end;
        align-items: flex-end;
    }
    
    /* 助手消息容器 */
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话历史和模型选择
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个乐于助人的AI助手，可以帮助用户进行肤质分析和护肤建议"}]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "glm-4-flash"

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# 图片转Base64函数
def image_to_base64(image):
    """将PIL图片转换为Base64编码"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

# 分析肤质的函数 - 支持全身皮肤分析
def analyze_skin_from_image(image, user_question=""):
    """使用GLM视觉模型分析全身皮肤状况（支持面部、腹部、背部等）"""
    img_base64 = image_to_base64(image)
    
    analysis_prompt = f"""请根据这张图片详细分析用户的皮肤状况，提供专业的护肤建议。

⚠️ 请注意：这是对全身皮肤（面部、腹部、背部、四肢等）的综合分析

请分析以下几个方面：
1. **皮肤区域位置**：识别图片中是面部、身体哪个部位（头部、躯干、四肢等）
2. **肤质类型判断**：油皮/干皮/混油/混干/敏感肌
3. **皮肤问题诊断**（重点识别）：
   - 痘痘/粉刺（黑头、白头、炎症痘）
   - 毛孔粗大程度
   - 细纹/皱纹
   - 皮疹（荨麻疹、湿疹表现等）
   - 泛红/潮红区域
   - 丘疹（小凸起）
   - 色素沉着/斑点
   - 皮肤纹理和粗糙度
   - 其他异常（脱皮、皮屑、炎症等）
4. **严重程度评估**：轻度/中度/重度
5. **护肤需求**：优先级和推荐方案
6. **产品建议**：针对性的护肤品推荐（含成分和用法）
7. **生活建议**：日常护肤建议和避坑指南

{f'用户补充信息：{user_question}' if user_question else ''}

请用专业但易懂的语言回答。如涉及皮肤病症状，请提醒用户咨询医生。"""
    
    stream = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": analysis_prompt
                    }
                ]
            }
        ],
        stream=True
    )
    
    return stream

# 顶部工具栏
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.session_state.selected_model = st.selectbox(
        "选择AI模型",
        ["glm-4-flash", "glm-4", "glm-3-turbo"],
        index=["glm-4-flash", "glm-4", "glm-3-turbo"].index(st.session_state.selected_model)
    )

with col2:
    if st.button("清空聊天记录", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": "你是一个乐于助人的AI助手，可以帮助用户进行肤质分析和护肤建议"}]
        st.session_state.uploaded_image = None
        st.rerun()

with col3:
    messages_export = [msg for msg in st.session_state.messages if msg["role"] != "system"]
    if messages_export:
        export_format = st.selectbox("导出格式", ["JSON", "文本"], key="export_format")
        
        if export_format == "JSON":
            export_data = json.dumps(messages_export, ensure_ascii=False, indent=2)
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            export_lines = []
            for msg in messages_export:
                role_name = "用户" if msg["role"] == "user" else "助手"
                export_lines.append(f"{role_name}: {msg['content']}\n")
            export_data = "\n".join(export_lines)
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        st.download_button(
            label="💾 下载记录",
            data=export_data,
            file_name=filename,
            mime="application/json" if export_format == "JSON" else "text/plain",
            use_container_width=True
        )

with col4:
    st.metric("消息数", len(st.session_state.messages) - 1)

st.divider()

# 免责声明
st.info("""
🏥 **免责声明**：本APP提供的皮肤分析结果是AI辅助分析，**不能替代医生的专业诊断和治疗建议**。
如果您的皮肤问题严重或持续恶化，请及时就医咨询专业医生。
""")

# 全身皮肤分析区域（可选）
st.subheader("📸 全身皮肤分析（可选）")
st.caption("支持分析：面部、颈部、腹部、背部、四肢等全身皮肤")
col_img, col_input = st.columns([1, 1])

with col_img:
    uploaded_file = st.file_uploader("上传照片（可选）", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的照片", use_container_width=True)
        st.session_state.uploaded_image = image

with col_input:
    image_question = st.text_area(
        "补充说明（可选）",
        placeholder="例如：\n- 这是我的背部皮肤\n- 最近长了很多痘痘\n- 皮肤很干又很痒\n- 这些红色斑点已经两周了...",
        height=120,
        key="image_question"
    )
    
    if st.button("🔍 分析皮肤", use_container_width=True):
        if st.session_state.uploaded_image is not None:
            st.session_state.messages.append({
                "role": "user",
                "content": f"请分析我上传的皮肤照片。{image_question if image_question else '请进行全面分析。'}"
            })
            
            with st.spinner("正在分析皮肤..."):
                stream = analyze_skin_from_image(st.session_state.uploaded_image, image_question)
                response = st.write_stream(stream)
            
            # 在回复中添加免责声明
            disclaimer = "\n\n" + "="*50 + "\n⚠️ **免责声明**：本分析结果仅供参考，不能替代医生诊断。如有严重皮肤问题，请咨询医生。"
            st.session_state.messages.append({"role": "assistant", "content": response + disclaimer})
            st.rerun()
        else:
            st.info("💡 提示：上传照片将获得更精准的皮肤分析，或者直接提问也可以哦~")

st.divider()

# 显示历史对话
st.subheader("💬 聊天记录")
for msg in st.session_state.messages[1:]:  # 跳过system提示词
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message"><div class="assistant-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

# 用户输入框
st.divider()
st.subheader("💭 提问")
if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-message"><div class="user-bubble">{prompt}</div></div>', unsafe_allow_html=True)

    with st.spinner("正在思考..."):
        stream = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=st.session_state.messages,
            stream=True
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
