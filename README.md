# 期中報告: 利用Google Colab訓練YOLO
## 👥 一、組員
* **11224132 蔡德鋒**
* **11224133 周栢正**

## 🎯 二、實作目的
本專案旨在結合 **YOLO 物件偵測演算法** 與 **Gradio 互動式 Web 介面**，建立一套具備即時辨識與結果視覺化的 AI 物件偵測系統。

詳細目標如下：

1. **自動化物件偵測與分類**：
 透過自訂資料集訓練 YOLO 模型，實現對特定目標（如：`milk`、`cup` 等）的精準定位與類別辨識。

2. **優化使用者操作體驗 (UI/UX)**：
 利用 Gradio 打造直覺的 Web 應用介面，讓使用者無需具備程式背景即可透過瀏覽器上傳圖片並即時取得辨識結果與數量統計。

3. **完整 AI 開發流程實踐**：
 整合自訂資料集建立（`data.yaml`）、模型訓練、評估與應用端部署（Gradio）的完整機器學習 pipeline。
## 🛠️三、使用工具

### 💻 開發環境 (Development Environment)
* **作業系統 / 平台**：Google Colab (GPU: NVIDIA T4)
* **開發語言**：Python 3.10+
* **使用工具**：Makesense (用來建立圖片標註類別) 

### 📦 使用工具與核心套件 (Tools & Libraries)
* **物件偵測模型**：Ultralytics YOLO (YOLOv8 / YOLOv11)
* **Web UI 框架**：Gradio
## 📂 四、檔案結構
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
## 🚀 五、執行步驟與流程

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
## 🖼️六、結果畫面

利用 Gradio 建立簡潔的 Web 應用介面，左側可上傳待測圖片，右側會即時回傳視覺化畫框結果與底下對應的物品名稱及總數統計。

<img width="1917" height="871" alt="螢幕擷取畫面 2026-10-27 011601" src="https://github.com/user-attachments/assets/ddaa74f4-888e-4fdf-9a50-976eebb41f82" />

<img width="857" height="442" alt="螢幕擷取畫面 2026-10-27 011727" src="https://github.com/user-attachments/assets/ee27cc95-bbd2-4d9a-bd4f-e84d205f3d10" />

底下列出辨識人、物品名稱

## ❓七、遇到問題與解決
我們主要遇到了以下兩項關鍵問題，並採取了對應的解決方案：

> [!WARNING]
> ### 1. 找不到資料集路徑 (`FileNotFoundError`)
* **問題現象**：執行訓練程式時，YOLO 拋出 `FileNotFoundError`，無法讀取 `images` 或 `labels` 資料夾。
* **原因分析**：Google Colab 的預設執行路徑（`/content/`）與 Google Drive 掛載後的實體路徑不符，導致相對路徑讀取失敗。

 *💡 解決方案*：
  * 在 `data.yaml` 中統一使用全絕對路徑：`/content/drive/MyDrive/my_dataset`。
  * 確保資料夾層級完全符合 YOLO 官方要求的標準結構（`images/train`, `labels/train`）。
<img width="982" height="276" alt="螢幕擷取畫面 2026-10-27 013549" src="https://github.com/user-attachments/assets/ed8fbd34-ddef-4016-8efb-d1d0e743abcf" />
<img width="1045" height="440" alt="螢幕擷取畫面 2026-10-27 013654" src="https://github.com/user-attachments/assets/ea90568f-343a-4a5b-a2ac-1bef5a206902" />
<img width="901" height="572" alt="螢幕擷取畫面 2026-10-27 013750" src="https://github.com/user-attachments/assets/24bc416a-1cc6-4714-b943-8940e373d6dc" />
<img width="887" height="642" alt="螢幕擷取畫面 2026-10-27 013821" src="https://github.com/user-attachments/assets/1cf74a9c-211b-4d83-a8f2-7f33d860d0f3" />

> [!WARNING]
> ### 2. 圖片與標註檔檔名不一致及數量不對等
> * **問題現象**：訓練時模型無法學習，或出現標註數量與圖片數量不相符的警告資訊。
> * **原因分析**：
>   1. `images` 資料夾內的圖片檔名（例如 `car.jpg`）與 `labels` 資料夾內的標註檔名（例如 `bus.txt`）未完全一致。
>   2. 訓練集與驗證集中的 `.jpg` 檔數量與 `.txt` 檔數量不相等（例如 8 個標註檔對應 7 張圖片）。
 
*💡 解決方案*：
* 重新清理並整理資料集，確保每一張圖片（如 `photo01.jpg`）都嚴格對應一個同名的標註檔（`photo01.txt`）。
* 確保 `images` 與 `labels` 內的檔案數量達到 **100% 1:1 完全配對**，使模型能正確關聯圖像與邊界框。
<img width="936" height="627" alt="螢幕擷取畫面 2026-10-27 014417" src="https://github.com/user-attachments/assets/29e33c0b-e171-408b-b3a9-e464fcba532a" />

## 💡 八、專案結論
>

本專案成功打造了一套端到端的 **YOLO AI 物件辨識系統**，實現了從原始影像資料處理至 Web 應用端部署的完整機器學習生命週期 (ML Lifecycle)。

### 🌟 成果

* **📦 完整的 AI Pipeline 建置**  
  涵蓋 Makesense 標註、YOLO 標準資料集建構 (`train`/`val` 拆分)、`data.yaml` 自動化生成，至模型訓練與驗證的全流程。

* **⚡ 輕量化與高效能推論**  
  採用 YOLO 輕量化模型，在 Google Colab GPU (T4) 資源下快速收斂，達到高精準度與即時推論速度。

* **🖥️ 低門檻的 Web 互動體驗**  
  結合 Gradio 打造簡潔直覺的雙欄 Web UI，使用者無需撰寫程式碼即可透過瀏覽器上傳圖片，並取得視覺化標註與物件數量統計結果。
