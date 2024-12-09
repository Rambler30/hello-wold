from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
import llm
import uvicorn
from llm import Qwen2VLModel
import config
from pydantic import BaseModel
from typing import Union, List
from pathlib import Path
import pdf
app = FastAPI()

llm = Qwen2VLModel(model_path="/gemini/code/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct")


class ImageUrlRequest(BaseModel):
    image_url: Union[str, List[str]]

# 发票
@app.post("/invoice/")
async def invoice(request: ImageUrlRequest):
    try:
        # 将上传的文件转换为图片
        image_url = request.image_url
        text_prompt = config.invoice
        # 使用 llm 进行 OCR
        if isinstance(image_url, str):
            image_url = ["file://"+image_url]
        elif isinstance(image_url, list):
            for i, p in enumerate(image_url):
                image_url[i] = "file://" + p
        else:
            return JSONResponse(content={"error": "image_url mast be string or list"})
        output = llm.infer(image_url, text_prompt)
        return JSONResponse(content={"text": output, "class": "发票"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# 申请表
@app.post("/application/")
async def application(request: ImageUrlRequest):
    try:
        # 将上传的文件转换为图片
        text_prompt = config.application
        image_url = request.image_url
        # 使用 llm 进行 OCR
        if isinstance(image_url, str):
            image_url = ["file://"+image_url]
        elif isinstance(image_url, list):
            for i, p in enumerate(image_url):
                image_url[i] = "file://" + p
        else:
            return JSONResponse(content={"error": "image_url mast be string or list"})
        output = llm.infer(image_url, text_prompt)
        output["class"] = "申请单"
        return JSONResponse(content={"text": output})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# 确认表
@app.post("/confirmation/")
async def confirmation(request: ImageUrlRequest):
    try:
        # 将上传的文件转换为图片
        text_prompt = config.confirmation
        image_url = request.image_url
        # 使用 llm 进行 OCR
        if isinstance(image_url, str):
            image_url = ["file://"+image_url]
        elif isinstance(image_url, list):
            for i, p in enumerate(image_url):
                image_url[i] = "file://" + p
        else:
            return JSONResponse(content={"error": "image_url mast be string or list"})
        output = llm.infer(image_url, text_prompt)
        output["class"] = "确认表"
        return JSONResponse(content={"text": output})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# 合同
@app.post("/contract/")
async def contract(request: ImageUrlRequest):
    try:
        # 将上传的文件转换为图片
        pdf_url = request.image_url
        text_prompt = config.contract
        # 使用 llm 进行 OCR
        save_path = "/gemini/code/contract/temp"
        img_url = pdf.analyze_contract(pdf_url, save_path)
        for i, p in enumerate(img_url):
            img_url[i] = "file://" + p
        output = llm.infer(img_url, text_prompt)
        output["class"] = "合同表"
        return JSONResponse(content={"text": output})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
