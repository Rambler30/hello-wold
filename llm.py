from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import config
from accelerate import Accelerator
import pdf

class Qwen2VLModel:
    def __init__(self, model_path: str, max_pixels: int = 1280 * 28 * 28):
        # accelerator = Accelerator()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        # self.model = accelerator.prepare(model)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            max_pixels=max_pixels
        )

    def infer(self, image_url: list, user_prompt: int, system_prompt: str = ""):
        """
        执行推理，返回生成的文本
        :param image_url: 多张图像路径，以列表形式存储   ["url1", "url2"]
        :param user_prompt: 用户输入的文本提示
        :return: 模型生成的文本列表
        """

        # 构建对话结构
        conversation = [{
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [],
            }]
        for p in image_url:
            type_d = {"type": "image"}
            type_d["image"] = p
            conversation[1]["content"].append(type_d)
        prompt_d = {"type": "text"}
        prompt_d["text"] = user_prompt
        conversation[1]["content"].append(prompt_d)

        # 生成文本提示并预处理输入
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        # 调用模型进行推理
        output_ids = self.model.generate(**inputs, max_new_tokens=32768)

        # 提取生成的新token（去除输入部分）
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        # 解码结果
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        t = output_text[0].strip("```json").strip("```")
        output_json = json.loads(t)
        return output_json
    

import time
if __name__ == '__main__':
    qwen2 = Qwen2VLModel(model_path="/gemini/code/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct")
    
    bool = True
    while(bool):
        url_list = []
        url_list.append("file:///gemini/code/image/1/ivc-591248005.jpg")
        start = time.time()
        t = qwen2.infer(image_url=url_list, user_prompt=config.invoice)
        print(time.time()-start)
        print(t)
