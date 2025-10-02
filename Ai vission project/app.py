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
import base64

# ================= Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Custom CSS =================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #141e30, #243b55);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .hero {
        text-align: center;
        padding: 1.5rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        color: #ff4b4b;
        margin-bottom: 0.5rem;
    }
    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff4b4b, #ff6f6f);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
        transition: 0.3s;
        border: none;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(45deg, #ff6f6f, #ff4b4b);
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
st.markdown("<div class='hero'><h1>🚀 AI Object Segmentation</h1><p>Cut out objects from images with smooth edges & style</p></div>", unsafe_allow_html=True)

# ================= Centered Banner Image =================
with open("img.png", "rb") as f:
    data = f.read()
b64 = base64.b64encode(data).decode()
st.markdown(
    f"""
    <div style='text-align:center; margin-top:20px;'>
        <img src='data:image/png;base64,{b64}' width='300'/>
        <p style='color:#ccc; margin-top:5px;'>AI Powered Segmentation</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= Sidebar Options =================
st.sidebar.header("⚙️ Options")
blur_strength = st.sidebar.slider("🎛️ Smoothness", 3, 21, 7, step=2)
use_transparent = st.sidebar.checkbox("Use Transparent Background")
tta_toggle = st.sidebar.checkbox("Use TTA for better quality", True)
dilate_iter = st.sidebar.slider("Mask Dilation", 0, 5, 1)

bg_type = st.sidebar.radio("Background Type", ["Solid Color","Gradient","Custom Image"])
gradient_colors = None  

if bg_type == "Solid Color":
    bg_color = st.sidebar.color_picker("🎨 Background Color", "#000000")
    bg_tuple = tuple(int(bg_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    custom_bg = None

elif bg_type == "Custom Image":
    bg_file = st.sidebar.file_uploader("Upload Background", type=["jpg","png"])
    custom_bg = Image.open(bg_file).convert("RGB") if bg_file else None
    bg_tuple = (0,0,0)

elif bg_type == "Gradient":
    color1 = st.sidebar.color_picker("🎨 Gradient Start", "#000000")
    color2 = st.sidebar.color_picker("🎨 Gradient End", "#ff0000")
    col1 = np.array([int(color1.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
    col2 = np.array([int(color2.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
    gradient_colors = (col1, col2)
    custom_bg = None
    bg_tuple = (0,0,0)

# ================= Mode Selection =================
mode = st.radio("Select Mode", ["Single Image", "Multiple Images"])

# ================= Single Image =================
if mode == "Single Image":
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="📷 Uploaded Image", width=250)

        if st.button("✨ Generate Image"):
            progress = st.progress(0)
            with st.spinner("AI is segmenting your image..."):
                for i in range(0,101,20):
                    time.sleep(0.2)
                    progress.progress(i)

                img_tensor, orig_size, _ = preprocess_image(img)
                mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)

                result = extract_object(img_np, pred_mask, bg_color=bg_tuple, transparent=use_transparent, custom_bg=custom_bg, gradient=gradient_colors)

            progress.empty()

            result_rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB) if result.shape[2]==4 else result

            tabs = st.tabs(["🖼️ Original vs Segmented", "↔️ Slider Comparison"])
            with tabs[0]:
                col1, col2 = st.columns(2)
                col1.image(img_np, caption="📷 Original", width=250)
                col2.image(result, caption="✨ Segmented", width=250)

                buf_seg = BytesIO(); Image.fromarray(result).save(buf_seg, format="PNG")
                st.download_button("📥 Download Segmented Image", buf_seg.getvalue(), "masked_image.png", "image/png")
                if use_transparent:
                    buf_trans = BytesIO(); Image.fromarray(result).save(buf_trans, format="PNG")
                    st.download_button("📥 Download Transparent PNG", buf_trans.getvalue(), "transparent.png", "image/png")

            img_small = cv2.resize(img_np, (250, int(250*img_np.shape[0]/img_np.shape[1])))
            result_bg = extract_object(img_np, pred_mask, bg_color=bg_tuple, transparent=False, custom_bg=custom_bg, gradient=gradient_colors)
            overlay_small = cv2.resize(result_bg, (250, int(250*result_bg.shape[0]/result_bg.shape[1])))
            with tabs[1]:
                image_comparison(img1=img_small, img2=overlay_small, label1="Original", label2="Segmented with BG")

# ================= Multiple Images =================
elif mode == "Multiple Images":
    uploaded_files = st.file_uploader("📤 Upload Multiple Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        images = [np.array(Image.open(f).convert("RGB")) for f in uploaded_files]
        cols = st.columns(len(images))
        for col, img_np in zip(cols, images):
            col.image(img_np, width=200)

        if st.button("✨ Segment All Images"):
            results = []
            progress = st.progress(0)
            for i, img_np in enumerate(images):
                img_tensor, orig_size, _ = preprocess_image(Image.fromarray(img_np))
                mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)
                result = extract_object(img_np, pred_mask, bg_color=bg_tuple, transparent=use_transparent, custom_bg=custom_bg, gradient=gradient_colors)
                results.append(result)
                progress.progress(int((i+1)/len(images)*100))
            progress.empty()

            cols = st.columns(len(results))
            for col, result in zip(cols, results):
                col.image(result, width=200)

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, result in enumerate(results):
                    buf = BytesIO()
                    Image.fromarray(result).save(buf, format="PNG")
                    zip_file.writestr(f"segmented_{idx+1}.png", buf.getvalue())
            st.download_button("📥 Download All Segmented Images", zip_buffer.getvalue(), "segmented_images.zip", "application/zip")
