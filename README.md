# 期中報告: 利用Google Colab訓練YOLO
## 一、組員
- 蔡德鋒 、周栢正

## 二、實作目的
本專案旨在結合 **YOLO 物件偵測演算法** 與 **Gradio 互動式 Web 介面**，建立一套具備即時辨識與結果視覺化的 AI 物件偵測系統。

詳細目標如下：

1. **自動化物件偵測與分類**：
 透過自訂資料集訓練 YOLO 模型，實現對特定目標（如：`milk`、`cup` 等）的精準定位與類別辨識。

2. **優化使用者操作體驗 (UI/UX)**：
 利用 Gradio 打造直覺的 Web 應用介面，讓使用者無需具備程式背景即可透過瀏覽器上傳圖片並即時取得辨識結果與數量統計。

3. **完整 AI 開發流程實踐**：
 整合自訂資料集建立（`data.yaml`）、模型訓練、評估與應用端部署（Gradio）的完整機器學習 pipeline。
## 三、使用工具
## 🛠️ 技術棧與開發環境 (Tech Stack & Environment)

### 💻 開發環境 (Development Environment)
* **作業系統 / 平台**：Google Colab (GPU: NVIDIA T4)
* **開發語言**：Python 3.10+
* **使用工具**：Makesense (建立標註類別) 

### 📦 使用工具與核心套件 (Tools & Libraries)
* **物件偵測模型**：Ultralytics YOLO (YOLOv8 / YOLOv11)
* **Web UI 框架**：Gradio
## 四、檔案結構
```text
MyDrive/
├── my_dataset/                # 資料集目錄
│   ├── images/
│   │   ├── train/            # 訓練集圖片
│   │   └── val/              # 驗證集圖片
│   └── labels/
│       ├── train/            # 訓練集標註檔 (.txt)
│       └── val/              # 驗證集標註檔 (.txt)
├── runs/                      # 模型訓練結果與權重檔
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt   # 最佳模型權重
├── data.yaml                  # YOLO 資料集設定檔
├── create_yaml.py             # 自動產生 data.yaml 腳本
├── train.py                   # 模型訓練主腳本
└── app.py                     # Gradio Web 介面啟動檔
```
## 五、執行步驟

🚀 執行步驟與流程 (Workflow & Steps)

---

### 1️⃣ GPU 檢查與基礎套件安裝

```python
import torch

# 檢查 GPU 狀態
if torch.cuda.is_available():
    print("✅ GPU 已成功啟用！")
    print("GPU 型號：", torch.cuda.get_device_name(0))
else:
    print("⚠️ 目前使用 CPU，建議前往選單切換至 GPU 執行階段！")

# 安裝核心套件
!pip install -q ultralytics gradio
```

### 2️⃣ 測試預訓練模型
```python
from ultralytics import YOLO
from IPython.display import display, Image
from pathlib import Path

# 載入預訓練模型
model = YOLO("yolov8n.pt")

# 使用測試圖片進行辨識
results = model.predict(
    source="[https://ultralytics.com/images/bus.jpg](https://ultralytics.com/images/bus.jpg)",
    save=True,
    conf=0.25
)

# 顯示辨識結果圖
image_path = Path(results[0].save_dir) / "bus.jpg"
display(Image(filename=str(image_path)))
```

### 3️⃣ 掛載 Google Drive 與建立資料集結構
```python
from ultralytics import YOLO
from IPython.display import display, Image
from pathlib import Path

# 載入預訓練模型
model = YOLO("yolov8n.pt")

# 使用測試圖片進行辨識
results = model.predict(
    source="[https://ultralytics.com/images/bus.jpg](https://ultralytics.com/images/bus.jpg)",
    save=True,
    conf=0.25
)

# 顯示辨識結果圖
image_path = Path(results[0].save_dir) / "bus.jpg"
display(Image(filename=str(image_path)))
```
### 4️⃣ 模型訓練 (Model Training)
```python
from ultralytics import YOLO

# 載入基礎預訓練權重
model = YOLO("yolov8n.pt")

# 開始訓練
results = model.train(
    data=f"{DATASET_PATH}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project=f"{DATASET_PATH}/runs",
    name="exp1"
)

print("🎉 模型訓練完成！")
```
### 5️⃣ 啟動 Gradio Web 互動介面
```python
import gradio as gr
from ultralytics import YOLO
from PIL import Image

# 載入訓練好的最佳模型
MODEL_PATH = f"{DATASET_PATH}/runs/exp1/weights/best.pt"
model = YOLO(MODEL_PATH)

def detect_image(image):
    if image is None:
        return None, "請上傳圖片。"

    results = model.predict(source=image, conf=0.25, imgsz=640, verbose=False)
    result = results[0]

    # 繪製辨識結果框
    annotated_image = result.plot()[:, :, ::-1]
    output_image = Image.fromarray(annotated_image)

    # 整理文字資訊
    if len(result.boxes) == 0:
        information = "沒有偵測到符合條件的物品。"
    else:
        names = result.names
        detected_items = [names[int(box.cls[0])] for box in result.boxes]
        
        information = f"共偵測到 {len(detected_items)} 個物件：\n\n"
        for idx, item in enumerate(detected_items, start=1):
            information += f"{idx}. {item}\n"

    return output_image, information

# 建立 Gradio UI
with gr.Blocks(title="YOLO AI 物件辨識系統", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 YOLO AI 物件辨識系統")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="上傳圖片")
            detect_button = gr.Button("🚀 開始辨識", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="辨識結果")
            output_text = gr.Textbox(label="詳細辨識資訊", lines=8)

    detect_button.click(
        fn=detect_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

# 啟動並產生公開連結
demo.launch(share=True)
```
## 六、結果畫面


<img width="857" height="442" alt="螢幕擷取畫面 2026-10-27 011727" src="https://github.com/user-attachments/assets/ee27cc95-bbd2-4d9a-bd4f-e84d205f3d10" />
