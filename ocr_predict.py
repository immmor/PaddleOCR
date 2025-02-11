from paddleocr import PaddleOCR  
import cv2  

# 创建 OCR 模型实例（lang参数指定识别
#ocr = PaddleOCR(det_model_dir='inference_model/ori_model/ch_PP-OCRv4_det_infer', rec_model_dir='inference_model/ori_model/ch_PP-OCRv4_rec_infer', use_angle_cls=False,use_gpu=True)
ocr = PaddleOCR(det_model_dir='inference_model/det/student', rec_model_dir='inference_model/rec', use_angle_cls=False,use_gpu=True)
# 读取图像  
image_path = 'test_img/1.png'  # 替换为你的图像路径  
img = cv2.imread(image_path)  

# 进行文字识别  
results = ocr.ocr(image_path, cls=True)  

# 输出识别结果  
for result in results:  
    for word_info in result:  
        # 打印识别的文本和坐标  
        print(f"文本: {word_info[1][0]}, 置信度: {word_info[1][1]}, 位置: {word_info[0]}")