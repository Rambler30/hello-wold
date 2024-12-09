import fitz  # PyMuPDF

def analyze_contract(file_path, save_path):
    pdf_document = fitz.open(file_path)
    # # 遍历每一页，提取图片
    # if file_type == 1:
    #     break_num = 3
    # elif file_type == 2:
    break_num = 5
    paths = []
    name = file_path.split("/")[-1].split(".")[0]
    for page_num in range(pdf_document.page_count):

        if page_num==break_num:
            break
        page = pdf_document.load_page(page_num)
        # 获取图片（如果页面有图片）
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            
            xref = img[0]  # 图像的xref
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]  # 图片的字节数据
            image_filename = name + f"_page_{page_num+1}_img_{img_index+1}.png"
            
            path = save_path + "/" + image_filename
            print(file_path) 
            
            paths.append(path)
             
            # 保存图片
            with open(path, "wb") as img_file:
                img_file.write(image_bytes)
                 
    return paths


if __name__ == "__main__":
    file_path = "/gemini/code/contract/广东电网有限责任公司信息中心电网管理平台电网规划管理应用建设项目信息系统开发实施合同.pdf"
    save_path = "/gemini/code/contract/temp"
    file_type = 1
    analyze_contract(file_path, save_path)
