import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from io import BytesIO
from streamlit_image_comparison import image_comparison
import zipfile
import time

# ================= Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Enhanced Custom CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* Root variables for smooth theme transitions */
:root {
    --transition-speed: 0.4s;
    --shadow-light: 0 8px 32px rgba(0,0,0,0.08);
    --shadow-medium: 0 12px 48px rgba(0,0,0,0.12);
    --shadow-heavy: 0 20px 60px rgba(0,0,0,0.15);
}

/* Global smooth transitions */
* {
    transition: background-color var(--transition-speed) ease,
                color var(--transition-speed) ease,
                border-color var(--transition-speed) ease,
                box-shadow var(--transition-speed) ease,
                transform 0.3s ease;
}

/* Light mode */
@media (prefers-color-scheme: light) {
    .stApp { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #f1f3f5 100%);
        color: #212529;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .hero { 
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid rgba(230,57,70,0.1);
        box-shadow: var(--shadow-medium);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(230,57,70,0.05) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    .hero h1 { 
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .hero p { 
        color: #6c757d;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 0.01em;
    }
    
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid rgba(0,0,0,0.06);
        box-shadow: 4px 0 24px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] { 
        background: #f1f3f5;
        color: #495057;
        border-radius: 12px 12px 0 0;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e9ecef;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(230,57,70,0.35) !important;
        transform: translateY(-2px);
    }
    
    .upload-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow-light);
        border: 2px dashed #dee2e6;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #e63946;
        box-shadow: var(--shadow-medium);
    }
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .stApp { 
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
        color: #e9ecef;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .hero { 
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border: 1px solid rgba(255,107,107,0.15);
        box-shadow: 0 12px 48px rgba(0,0,0,0.6);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,107,107,0.08) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    .hero h1 { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .hero p { 
        color: #adb5bd;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 0.01em;
    }
    
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
        box-shadow: 4px 0 24px rgba(0,0,0,0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] { 
        background: #2a2a2a;
        color: #adb5bd;
        border-radius: 12px 12px 0 0;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #333333;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(255,107,107,0.4) !important;
        transform: translateY(-2px);
    }
    
    .upload-section {
        background: #1a1a1a;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 2px dashed #333333;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #ff6b6b;
        box-shadow: 0 12px 48px rgba(0,0,0,0.6);
    }
}

@keyframes pulse {
    0%, 100% { transform: scale(1) rotate(0deg); opacity: 1; }
    50% { transform: scale(1.1) rotate(5deg); opacity: 0.8; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

/* Enhanced Buttons */
.stButton > button {
    background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
    color: white !important;
    border-radius: 16px;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.85rem 2rem;
    border: none;
    box-shadow: 0 6px 24px rgba(230,57,70,0.3);
    position: relative;
    overflow: hidden;
    letter-spacing: 0.02em;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s ease;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {
    transform: translateY(-3px) scale(1.02);
    background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
    box-shadow: 0 12px 32px rgba(230,57,70,0.45);
    color: white !important;
}

/* Download Buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #20c997 0%, #38d9a9 100%);
    color: white !important;
    border-radius: 14px;
    font-weight: 600;
    padding: 0.7rem 1.8rem;
    border: none;
    box-shadow: 0 4px 16px rgba(32,201,151,0.3);
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(32,201,151,0.4);
    background: linear-gradient(135deg, #38d9a9 0%, #51cf66 100%);
    color: white !important;
}

/* Enhanced Images */
img {
    border-radius: 16px;
    box-shadow: var(--shadow-medium);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

img:hover {
    transform: scale(1.02);
    box-shadow: var(--shadow-heavy);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #e63946, #ff6b6b, #ff8787);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

/* Radio buttons */
.stRadio > label {
    font-weight: 600;
    font-size: 1.05rem;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #e63946, #ff6b6b);
}

/* Sidebar header */
.css-1d391kg, [data-testid="stSidebarNav"] {
    padding-top: 2rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 16px;
    padding: 1.5rem;
}

/* Checkboxes */
.stCheckbox {
    font-weight: 500;
}

/* Info messages */
.stAlert {
    border-radius: 12px;
    box-shadow: var(--shadow-light);
}

/* Spinner */
.stSpinner > div {
    border-top-color: #e63946 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= Load Model =================
@st.cache_resource
def load_model():
    num_classes = 2
    model_path = "deeplabv3_final.pth"
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    old_cls = model.classifier
    model.classifier = nn.Sequential(
        old_cls[0], old_cls[1], old_cls[2], nn.Dropout(0.3),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, num_classes, kernel_size=1)
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model

model = load_model()

# ================= Preprocess =================
def preprocess_image(img, long_side=256):
    img = np.array(img)
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    img_tensor = torch.tensor(img_resized/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.to(device), (h, w), img

# ================= TTA Prediction =================
def tta_predict(img_tensor):
    aug_list = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, k=1, dims=[2,3]),
        lambda x: torch.rot90(x, k=3, dims=[2,3])
    ]
    outputs = []
    with torch.no_grad():
        for f in aug_list:
            t = f(img_tensor)
            out = model(t)['out']
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out,[3])
                elif f == aug_list[2]: out = torch.flip(out,[2])
                elif f == aug_list[3]: out = torch.rot90(out,k=3,dims=[2,3])
                elif f == aug_list[4]: out = torch.rot90(out,k=1,dims=[2,3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

# ================= Post-process mask =================
def postprocess_mask(mask_tensor, orig_size, blur_strength=5):
    mask = torch.argmax(mask_tensor, dim=1).squeeze().cpu().numpy()
    mask = cv2.resize(mask.astype(np.uint8), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)
    _, smooth_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    return (smooth_mask // 255).astype(np.uint8)

# ================= Extract Object =================
def extract_object(img, mask, bg_color=(0,0,0), transparent=False, custom_bg=None, gradient=None):
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / (mask.max() + 1e-8)
    if transparent:
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA).astype(np.float32)
        rgba[..., 3] = (mask.squeeze() * 255).astype(np.uint8)
        return rgba.astype(np.uint8)
    else:
        if gradient is not None:
            h, w = img.shape[:2]
            col1, col2 = gradient
            bg_resized = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(h):
                alpha = i / max(h-1, 1)
                row_color = (1 - alpha) * col1.astype(np.float32) + alpha * col2.astype(np.float32)
                bg_resized[i, :, :] = row_color
        elif custom_bg is not None:
            bg_resized = cv2.resize(np.array(custom_bg), (img.shape[1], img.shape[0])).astype(np.float32)
        else:
            bg_resized = np.full_like(img, bg_color, dtype=np.float32)
        result = img.astype(np.float32) * mask + bg_resized.astype(np.float32) * (1 - mask)
        return result.astype(np.uint8)

# ================= Hero Section =================
st.markdown("""
<div class='hero'>
    <h1>✨ AI Object Segmentation</h1>
    <p>Transform your images with precision AI-powered object extraction</p>
</div>
""", unsafe_allow_html=True)

# === Centered Banner Image ===
st.markdown(
    """
    <div style='text-align: center; margin-top: 20px;'>
        <img src='img.png' width='320' alt='AI Powered Segmentation'/>
        <p style='margin-top: 5px; font-style: italic;'>AI Powered Segmentation</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= Sidebar Options =================
st.sidebar.markdown("### ⚙️ Segmentation Settings")
st.sidebar.markdown("---")

with st.sidebar.expander("🎨 Quality Settings", expanded=True):
    tta_toggle = st.checkbox("🔄 High Quality Mode (TTA)", True, help="Uses Test-Time Augmentation for better results")
    blur_strength = st.slider("✨ Edge Smoothness", 3, 21, 7, step=2, help="Higher values = softer edges")
    dilate_iter = st.slider("📏 Mask Expansion", 0, 5, 1, help="Expand or shrink the mask")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Background Settings")

use_transparent = st.sidebar.checkbox("🔲 Transparent Background", help="Create PNG with transparency")
bg_type = st.sidebar.radio("Background Style", ["Solid Color","Gradient","Custom Image"])

gradient_colors = None  

if bg_type == "Solid Color":
    bg_color = st.sidebar.color_picker("🎨 Background Color", "#000000")
    bg_tuple = tuple(int(bg_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    custom_bg = None
elif bg_type == "Custom Image":
    bg_file = st.sidebar.file_uploader("📸 Upload Background Image", type=["jpg","png"])
    custom_bg = Image.open(bg_file).convert("RGB") if bg_file else None
    bg_tuple = (0,0,0)
elif bg_type == "Gradient":
    col1_exp = st.sidebar.columns(2)
    with col1_exp[0]:
        color1 = st.color_picker("Start", "#000000")
    with col1_exp[1]:
        color2 = st.color_picker("End", "#ff0000")
    col1 = np.array([int(color1.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
    col2 = np.array([int(color2.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
    gradient_colors = (col1, col2)
    custom_bg = None
    bg_tuple = (0,0,0)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** High Quality Mode works best for complex objects!")

# ================= Mode Selection =================
st.markdown("### 🖼️ Select Processing Mode")
mode = st.radio("Select Processing Mode", ["Single Image", "Batch Processing"], horizontal=True, label_visibility="collapsed")

# ================= Single Image =================
if mode == "Single Image":
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload Your Image", type=["jpg","jpeg","png"], help="Supported formats: JPG, PNG")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            process_btn = st.button("✨ Process Image", use_container_width=True)
        
        if process_btn:
            progress = st.progress(0, text="🔄 Initializing AI model...")
            
            with st.spinner("🎨 AI is working its magic..."):
                for i in range(0,101,20):
                    if i == 20:
                        progress.progress(i, text="🔍 Analyzing image...")
                    elif i == 40:
                        progress.progress(i, text="🎯 Detecting objects...")
                    elif i == 60:
                        progress.progress(i, text="✂️ Segmenting...")
                    elif i == 80:
                        progress.progress(i, text="🎨 Refining edges...")
                    else:
                        progress.progress(i, text="✅ Finalizing...")
                    time.sleep(0.2)

                img_tensor, orig_size, _ = preprocess_image(img)
                mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)

                result = extract_object(img_np, pred_mask, bg_color=bg_tuple, transparent=use_transparent, custom_bg=custom_bg, gradient=gradient_colors)

            progress.empty()
            st.success("✅ Segmentation Complete!")
            st.markdown("---")

            tabs = st.tabs(["🖼️ Side by Side", "↔️ Interactive Comparison"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📷 Original**")
                    st.image(img_np, use_container_width=True)
                with col2:
                    st.markdown("**✨ Segmented**")
                    st.image(result, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📥 Download Results")
                
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    buf_seg = BytesIO()
                    Image.fromarray(result).save(buf_seg, format="PNG")
                    st.download_button("💾 Download Segmented", buf_seg.getvalue(), "segmented_image.png", "image/png", use_container_width=True)
                
                with dl_col2:
                    if use_transparent:
                        buf_trans = BytesIO()
                        Image.fromarray(result).save(buf_trans, format="PNG")
                        st.download_button("💾 Download Transparent", buf_trans.getvalue(), "transparent.png", "image/png", use_container_width=True)

            with tabs[1]:
                img_small = cv2.resize(img_np, (500, int(500*img_np.shape[0]/img_np.shape[1])))
                result_bg = extract_object(img_np, pred_mask, bg_color=bg_tuple, transparent=False, custom_bg=custom_bg, gradient=gradient_colors)
                overlay_small = cv2.resize(result_bg, (500, int(500*result_bg.shape[0]/result_bg.shape[1])))
                image_comparison(img1=img_small, img2=overlay_small, label1="Original", label2="Segmented")

# ================= Multiple Images =================
elif mode == "Batch Processing":
    st.markdown("---")
    uploaded_files = st.file_uploader("📤 Upload Multiple Images", type=["jpg","jpeg","png"], accept_multiple_files=True, help="Process multiple images at once")
    
    if uploaded_files:
        st.markdown(f"### 📸 Uploaded {len(uploaded_files)} images")
        
        images = [np.array(Image.open(f).convert("RGB")) for f in uploaded_files]
        cols = st.columns(min(len(images), 4))
        for idx, (col, img_np) in enumerate(zip(cols, images[:4])):
            col.image(img_np, caption=f"Image {idx+1}", use_container_width=True)
        
        if len(images) > 4:
            st.info(f"📋 {len(images)-4} more images ready for processing...")

        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            process_all = st.button("✨ Process All Images", use_container_width=True)
        
        if process_all:
            results = []
            progress = st.progress(0, text="🚀 Starting batch processing...")
            
            for i, img_np in enumerate(images):
                progress.progress(int((i)/len(images)*100), text=f"🎨 Processing image {i+1} of {len(images)}...")
                
                img_tensor, orig_size, _ = preprocess_image(Image.fromarray(img_np))
                mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)
                result = extract_object(img_np, pred_mask, bg_color=bg_tuple, transparent=use_transparent, custom_bg=custom_bg, gradient=gradient_colors)
                results.append(result)
            
            progress.progress(100, text="✅ All images processed!")
            time.sleep(0.5)
            progress.empty()
            
            st.success(f"✅ Successfully processed {len(results)} images!")
            st.markdown("---")
            st.markdown("### 🎉 Results")

            cols = st.columns(min(len(results), 4))
            for idx, (col, result) in enumerate(zip(cols, results[:4])):
                col.image(result, caption=f"Result {idx+1}", use_container_width=True)
            
            if len(results) > 4:
                with st.expander(f"👁️ View all {len(results)} results"):
                    remaining_cols = st.columns(4)
                    for idx, result in enumerate(results[4:], 5):
                        remaining_cols[(idx-5) % 4].image(result, caption=f"Result {idx}", use_container_width=True)

            st.markdown("---")
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, result in enumerate(results):
                    buf = BytesIO()
                    Image.fromarray(result).save(buf, format="PNG")
                    zip_file.writestr(f"segmented_{idx+1}.png", buf.getvalue())
            
            col_dl1, col_dl2, col_dl3 = st.columns([1,1,1])
            with col_dl2:
                st.download_button("📦 Download All as ZIP", zip_buffer.getvalue(), "segmented_images.zip", "application/zip", use_container_width=True)
