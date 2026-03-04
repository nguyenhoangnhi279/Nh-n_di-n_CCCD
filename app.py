import cv2
import re
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def clean_extracted_data(class_name, text):
    text = text.strip()
    if class_name == 'id_number':
        text = re.sub(r'\D', '', text) 
    elif class_name == 'dob':
        text = text.replace('-', '/').replace('.', '/') 
    elif class_name == 'name':
        text = text.upper() 
    return text

try:
    yolo_model = YOLO('best.pt')
except Exception as e:
    print(f"Lỗi: {e}")

config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu' 
config['cnn']['pretrained'] = False
ocr_model = Predictor(config)

def process_id_card(input_image):
    if input_image is None:
        return None, "Lỗi: Vui lòng tải ảnh lên"

    img_height, img_width, _ = input_image.shape
    output_image = input_image.copy()
    
    results = yolo_model(input_image)
    extracted_info = {}

    if len(results[0].boxes) == 0:
        return output_image, "Không tìm thấy thông tin."

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]

        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, class_name.upper(), (x1, max(y1 - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        padding = 5
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(img_width, x2 + padding)
        py2 = min(img_height, y2 + padding)

        cropped_img = input_image[py1:py2, px1:px2]
        pil_img = Image.fromarray(cropped_img)

        raw_text = ocr_model.predict(pil_img)
        final_text = clean_extracted_data(class_name, raw_text)
        
        extracted_info[class_name] = final_text

    result_text = ""
    name_mapping = {
        'id_number': 'Số ID',
        'name': 'Họ và tên',
        'dob': 'Ngày sinh',
        'address': 'Địa chỉ'
    }
    
    for key, value in extracted_info.items():
        vn_name = name_mapping.get(key, key.upper())
        result_text += f"▪️ {vn_name}: {value}\n\n"

    return output_image, result_text.strip()

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'> HỆ THỐNG TRÍCH XUẤT THÔNG TIN CĂN CƯỚC CÔNG DÂN</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="numpy", label="Tải ảnh CCCD lên đây")
            btn_submit = gr.Button("Bắt đầu", variant="primary")
            
        with gr.Column(scale=1):
            img_output = gr.Image(label="Ảnh đã nhận diện vùng chữ")
            text_output = gr.Textbox(label="Kết quả", lines=8)
            
    btn_submit.click(
        fn=process_id_card, 
        inputs=img_input, 
        outputs=[img_output, text_output] 
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())