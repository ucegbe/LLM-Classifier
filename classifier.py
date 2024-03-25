import boto3
from botocore.config import Config
import shutil
import os
import fitz
from textractor import Textractor
from textractor.visualizers.entitylist import EntityList
from textractor.data.constants import TextractFeatures
from PIL import Image
from io import BytesIO
import io
import pandas as pd
from botocore.exceptions import ClientError
import time
# import concurrent.futures
import json
import streamlit as st
from boto3.dynamodb.conditions import Key 
config = Config(
    read_timeout=600,
    retries = dict(
        max_attempts = 5 ## Handle retries
    )
)
import re
import base64
from streamlit.runtime.uploaded_file_manager import UploadedFile
st.set_page_config(layout="wide")
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'token' not in st.session_state:
    st.session_state['token'] = 0
if 'chat_hist' not in st.session_state:
    st.session_state['chat_hist'] = []
if 'page_summ' not in st.session_state:
    st.session_state['page_summ'] = ""
if 'extraction' not in st.session_state:
    st.session_state['extraction'] = ""
if 'cost' not in st.session_state:
    st.session_state['cost'] = 0
    
# Read credentials
with open('config.json') as f:
    config_file = json.load(f)
# pricing info
with open('pricing.json') as f:
    pricing_file = json.load(f)
    
BUCKET=config_file["Bucket_Name"]
OUTPUT_TOKEN=config_file["max-output-token"]
S3 = boto3.client('s3')
TEXTRACT_RESULT_CACHE_PATH=config_file["textract_result_path"]
REGION=config_file["bedrock-region"]
bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name=REGION,config=config)
PREFIX=config_file["s3_path_prefix"]

def bedrock_claude_(params,system_message, prompt,model_id,image_path=None, handler=None):
    content=[]
    if image_path:  

        if not isinstance(image_path, list):
            image_path=[image_path]      
        for img in image_path:
            match = re.match("s3://(.+?)/(.+)", img)
            image_name=os.path.basename(img)
            _,ext=os.path.splitext(image_name)
            if "jpg" in ext: ext=".jpeg"                        
            if match:
                bucket_name = match.group(1)
                key = match.group(2)    
                obj = S3.get_object(Bucket=bucket_name, Key=key)
                base_64_encoded_data = base64.b64encode(obj['Body'].read())
                base64_string = base_64_encoded_data.decode('utf-8')
            content.extend([{"type":"text","text":image_name},{
              "type": "image",
              "source": {
                "type": "base64",
                "media_type": f"image/{ext.lower().replace('.','')}",
                "data": base64_string
              }
            }])
    content.append({
        "type": "text",
        "text": prompt
            })
    # st.write(content)
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2500,
        "temperature": 0.5,
        "system":system_message,
        "messages": [    
            {
                "role": "user",
                "content": content
            }
        ]
    }

    prompt = json.dumps(prompt)
    response = bedrock_runtime.invoke_model_with_response_stream(body=prompt, modelId=model_id, accept="application/json", contentType="application/json")
    answer=bedrock_streemer(params,response, handler) 
    return answer



def bedrock_streemer(params,response, handler):
    stream = response.get('body')
    answer = ""    
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if  chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                if "delta" in chunk_obj:                    
                    delta = chunk_obj['delta']
                    if "text" in delta:
                        text=delta['text']                       
                        answer+=str(text)       
                        handler.markdown(answer.replace("$","USD ").replace("%", " percent"))                        
                if "amazon-bedrock-invocationMetrics" in chunk_obj:
                    st.session_state['input_token'] = chunk_obj['amazon-bedrock-invocationMetrics']['inputTokenCount']
                    st.session_state['output_token'] =chunk_obj['amazon-bedrock-invocationMetrics']['outputTokenCount']
                    pricing=st.session_state['input_token']*pricing_file[f"anthropic.{params['model']}"]["input"]+st.session_state['output_token'] *pricing_file[f"anthropic.{params['model']}"]["output"]
                    st.session_state['cost']+=pricing             
    return answer

def _invoke_bedrock_with_retries(params, chat_template, question, model_id, image_path, handler):
    max_retries = 5
    backoff_base = 2
    max_backoff = 3  # Maximum backoff time in seconds
    retries = 0
    while True:
        try:
            response = bedrock_claude_(params, chat_template, question, model_id, image_path, handler)
            return response
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            else:
                # Some other API error, rethrow
                raise e

@st.cache_data
def process_and_upload_files_to_s3(uploaded_files, s3_bucket, s3_prefix):
    """
    Uploads PDF and image files to an Amazon S3 bucket.
    Args:
        uploaded_files (list): A list of file-like objects representing the uploaded files.
        s3_bucket (str): The name of the Amazon S3 bucket to upload the files to.
        s3_prefix (str): The prefix to use for the uploaded file names in the S3 bucket.

    Returns:
        list: A list of dictionaries, where each dictionary contains information about the uploaded files.
              The dictionary has the following keys:
                - 'file_name': The name of the uploaded file.
                - 'file_paths': A list of S3 paths for the uploaded files.
    """
    s3_client = boto3.client('s3')
    upload_info = []
    image_paths = []
    for file in uploaded_files:
        file_bytes=s3_client.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{file}")['Body'].read()
        file_name=file
        if file_name.lower().endswith('.pdf'):            
            pdf_file = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")            
            for page_index in range(len(pdf_file)):
                # Select the page
                page = pdf_file[page_index]                
                # Get the page dimensions in PDF units (1/72 inch)
                page_rect = page.rect
                # Convert the dimensions from PDF units to inches
                page_width_inches = page_rect.width / 72
                page_height_inches = page_rect.height / 72
                # Calculate the maximum DPI to keep pixel dimensions below 1500
                max_dpi = min(1500 / page_width_inches, 1500 / page_height_inches)
                # Render the page as a PyMuPDF Image object
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), dpi=round(max_dpi))
                # Convert the PyMuPDF Image object to bytes
                image_bytes = pix.tobytes()
                # Construct the image file name
                image_filename = f"{s3_prefix}/{file_name.replace('.pdf', '')}-page-{page_index+1}.jpeg"
                # Upload the image to S3
                s3_client.put_object(Bucket=s3_bucket, Key=image_filename, Body=image_bytes)
                image_paths.append(f"s3://{s3_bucket}/{image_filename}")   
            pdf_file.close()
        else:
            # Assume it's an image file
            image_filename = f"{s3_prefix}/{file_name}"
            # Upload the image to S3
            s3_client.upload_fileobj(io.BytesIO(file_bytes), s3_bucket, image_filename)
            image_path = f"s3://{s3_bucket}/{image_filename}"
            image_paths.append(               
                image_path
           )
    return image_paths

def query_llm(params, handler):
    if not params["possible_labels"]:
        st.error("Please upload manifest file containing possible label set with their description. Enter those labels and description in a .txt file and upload.\n\n Example:\n\nreceipt : goods payment receipt\n\n license : US state ID\n\n...")
        st.stop()
    image_path=[]
    claude3=False
    model='anthropic.'+params['model']
    if "sonnet" in model or "haiku" in model:
        model+="-20240229-v1:0" if "sonnet" in model else "-20240307-v1:0"
        claude3=True
    with open(f"prompt/system.txt","r") as f:
        system_template=f.read()
    if "doc" in params:
        with open(f"prompt/sorter.txt","r") as f:
            prompt=f.read()
        prompt=prompt.replace("{doc}",params['doc']).replace("{label}",params['possible_labels'].read().decode())
        
    elif "image_path" in params:
        if len(params["image_path"])>20:
            st.error("Document image count is > 20, please reduce number of documents or select Textract to process documents\n\n When using Claude 3 as document processor, all pdf pages are counted as an image and will likely exceed the 20 threshold.")
            st.stop()
        with open(f"prompt/sorter_image.txt","r") as f:
            prompt=f.read()
        prompt=prompt.replace("{label}",params['possible_labels'].read().decode())
        image_path=params["image_path"]
    response=_invoke_bedrock_with_retries(params,system_template, prompt, model, image_path, handler)
    return response

def is_pdf(file_bytes):
    """
    Checks if the given bytes represent a PDF file.
    Args:
        file_bytes (bytes): The bytes to check.
    Returns:
        bool: True if the bytes represent a PDF file, False otherwise.
    """
    # PDF file signature is "%PDF"
    pdf_signature = b"%PDF"
    return file_bytes.startswith(pdf_signature)

def is_image(file_bytes):
    """
    Checks if the given bytes represent an image file.
    Args:
        file_bytes (bytes): The bytes to check.
    Returns:
        bool: True if the bytes represent an image file, False otherwise.
    """
    # Common image file signatures
    image_signatures = {
        b"\xff\xd8": "jpeg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38\x37\x61": "gif",
        b"\x47\x49\x46\x38\x39\x61": "gif",
        b"\x42\x4d": "bmp",
    }

    for signature, _ in image_signatures.items():
        if file_bytes.startswith(signature):
            return True

    return False

def get_s3_keys(prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    keys=""
    if "Contents" in response:
        keys = []
        for obj in response['Contents']:
            key = obj['Key']
            name = key[len(prefix):]
            keys.append(name)
    return keys
     

def page_summary(params):
    """
    This action takes the entire rendered document page as context for the following LLM actions below.
    """
    label_per_page=params["label_per_page"]
    import time
    pdf_file=""

    item_cache={}
    s3 = boto3.client('s3')
    if params['file_item'] and params["processor"] == "Textract":
        params['doc']=""
        if isinstance(params['file_item'], list):
            doc_list=[]
            if all(isinstance(file, UploadedFile) for file in params['file_item']):
                for file in params['file_item']:                   
                    pdf_name=file.name
                    pdf_bytes=file.read()
                    item_cache[pdf_name]=pdf_bytes
                    S3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{pdf_name}", Body=pdf_bytes)
                    doc_list.append(pdf_name)
                doc_file_names=[f"s3://{BUCKET}/{PREFIX}/{x}" for x in doc_list]
                if len(doc_list)==1:                                            
                    params['doc']+=get_text_ocr_(doc_file_names[0], True if label_per_page  else False)
                else:
                    params['doc']=""
                    for file in doc_file_names:
                        params['doc']+=get_text_ocr_(file, True if label_per_page  else False)
                params['file_item']=doc_list
                
    elif params['file_item'] and params["processor"] == "Claude3":
        if isinstance(params['file_item'], list):
            doc_list=[]
            if all(isinstance(file, UploadedFile) for file in params['file_item']):
                for file in params['file_item']:
                    pdf_name=file.name
                    pdf_bytes=file.read()
                    item_cache[pdf_name]=pdf_bytes
                    S3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{pdf_name}", Body=pdf_bytes)
                    doc_list.append(pdf_name)               
                if len(doc_list)==1:            
                    params['image_path']=process_and_upload_files_to_s3(doc_list, BUCKET, PREFIX) 
                else:
                    params['image_path']=process_and_upload_files_to_s3(doc_list, BUCKET, PREFIX)          
                params['file_item']=doc_list
                
                
                
    if item_cache:
        colm1,colm2=st.columns([1,1])
        page_count=len(item_cache.keys())
        with colm1:
            if page_count==1:
                filee=item_cache[list(item_cache.keys())[0]]
                st.markdown(f"##### :red[{list(item_cache.keys())[0]}]")
                if is_pdf(filee):                
                    base64_pdf = base64.b64encode(filee).decode('utf-8')
                    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="800" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                elif  is_image(filee):
                    img = Image.open(io.BytesIO(filee)) 
                    st.image(img)
            else:
                col1, col2, col3 = st.columns(3)
                # Buttons
                if col1.button("Previous", key="prev_page"):
                    st.session_state.page_slider=max(st.session_state.page_slider-1,0)
                if col3.button("Next", key="next_page"):
                    st.session_state.page_slider=min(st.session_state.page_slider+1,page_count-1)
                # Page slider
                col2.slider("Page Slider", min_value=0, max_value=page_count-1, key="page_slider")
                # Rendering pdf page 
                st.markdown(f"##### :red[{list(item_cache.keys())[st.session_state.page_slider]}]")
                # col2.header(f":blue[{list(item_cache.keys())[st.session_state.page_slider]}]")
                filee=item_cache[list(item_cache.keys())[st.session_state.page_slider]]
                if is_pdf(filee):                
                    base64_pdf = base64.b64encode(filee).decode('utf-8')
                    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="800" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                elif  is_image(filee):
                    img = Image.open(io.BytesIO(filee)) 
                    st.image(img)

        with colm2:            
            tab2,_= st.tabs(["**Classifier**","  "])     
            with tab2.container(height=500,border=False):
                if st.button('Classify',type="primary",key='calssify'):  
                    container_tab=st.empty()
                    summary=query_llm(params, container_tab)         
                    st.session_state['page_summ']=summary 
                else:
                    container_tab=st.empty()
                if "```" not in st.session_state['page_summ']:
                    st.session_state['page_summ']=f'```json\n{st.session_state["page_summ"]}'
                container_tab.markdown(st.session_state["page_summ"].replace("$", "\$"),unsafe_allow_html=True)
  
                    
                
                

@st.cache_data
def get_text_ocr_(file,label_per_page):    

    file_base_name=os.path.basename(file)+".txt"               
    _, file_ext = os.path.splitext(file)    
    doc_id=os.path.basename(file)
    extractor = Textractor(region_name="us-east-1")    
    image_extensions = ('.jpg', '.png', '.jpeg')
    if file_ext.lower().endswith(image_extensions):
        if [x for x in get_s3_keys(f"{TEXTRACT_RESULT_CACHE_PATH}/") if file_base_name == x]:                  
            response = S3.get_object(Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{file_base_name}")            
            text = response['Body'].read()  
            return f"<{os.path.basename(file)}>\n{text}\n</{os.path.basename(file)}>\n"     
        else:
            document = extractor.analyze_document(
                file_source=file,
                features=[TextractFeatures.LAYOUT,TextractFeatures.TABLES,TextractFeatures.FORMS],
                save_image=False,
            )
            from textractor.data.text_linearization_config import TextLinearizationConfig
            configs = TextLinearizationConfig(
                hide_figure_layout=True,
                hide_header_layout=False,
                table_prefix="<table>",
                table_suffix="</table>",
                hide_footer_layout=True,
                hide_page_num_layout=False,    
            )            
            S3.put_object(Body=document.get_text(config=configs), Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{doc_id}.txt")             
            return f"<{os.path.basename(file)}>\n{document.get_text(config=configs)}\n</{os.path.basename(file)}>\n"     
    elif file_ext.lower().endswith('.pdf'):
        if label_per_page:
            pdf_file=S3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{doc_id}")['Body'].read()
            pdf_document = fitz.open("pdf",pdf_file)  
            num_pages = len(pdf_document)
            pdf_document.close()
            doc=""
            if [x for x in get_s3_keys(f"{TEXTRACT_RESULT_CACHE_PATH}/") if doc_id+f"_Page1.txt" in x]:  
                for page_ids in range(num_pages):
                    doc_id=os.path.basename(file)+f"_Page{str(page_ids+1)}.txt"  
                    response = S3.get_object(Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{doc_id}")
                    text = response['Body'].read()
                    doc+=f"<{os.path.basename(file)}_Page{str(page_ids+1)}>\n{text}\n</{os.path.basename(file)}_Page{str(page_ids+1)}>\n"
                return doc
            else:
                document = extractor.start_document_analysis(
                    file_source=file,
                    features=[TextractFeatures.LAYOUT,TextractFeatures.TABLES,TextractFeatures.FORMS],
                    save_image=False,
                )
                from textractor.data.text_linearization_config import TextLinearizationConfig
                configs = TextLinearizationConfig(
                    hide_figure_layout=True,
                    hide_header_layout=False,
                    table_prefix="<table>",
                    table_suffix="</table>",
                    hide_footer_layout=True,
                    hide_page_num_layout=False,    
                )           
                for page_ids in range(len(document.pages)):
                    doc_id=os.path.basename(file)+f"_Page{str(page_ids+1)}.txt"      
                    S3.put_object(Body=document.pages[page_ids].get_text(config=configs), Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{doc_id}") 
                    doc+=f"<{os.path.basename(file)}_Page{str(page_ids+1)}>\n{document.pages[page_ids].get_text(config=configs)}\n</{os.path.basename(file)}_Page{str(page_ids+1)}>\n"
                return doc
               
        else:
            file_base_name=os.path.basename(file)+".txt"
            if [x for x in get_s3_keys(f"{TEXTRACT_RESULT_CACHE_PATH}/") if file_base_name in x]:     
                response = S3.get_object(Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{file_base_name}")
                text = response['Body'].read()
                return f"<{os.path.basename(file)}>\n{text}\n</{os.path.basename(file)}>\n"     
            else:
                document = extractor.start_document_analysis(
                    file_source=file,
                    features=[TextractFeatures.LAYOUT,TextractFeatures.TABLES,TextractFeatures.FORMS],
                    save_image=False,
                )
                from textractor.data.text_linearization_config import TextLinearizationConfig
                configs = TextLinearizationConfig(
                    hide_figure_layout=True,
                    hide_header_layout=False,
                    table_prefix="<table>",
                    table_suffix="</table>",
                    hide_footer_layout=True,
                    hide_page_num_layout=False,    
                )            
                S3.put_object(Body=document.get_text(config=configs), Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{doc_id}.txt")               
                return f"<{os.path.basename(file)}>\n{document.get_text(config=configs)}\n</{os.path.basename(file)}>\n"     
            
        
        
        
def app_sidebar():
    with st.sidebar:
        st.metric(label="Bedrock Session Cost", value=f"${round(st.session_state['cost'],2)}") 
        st.write("-----")       
        models=[ 'claude-3-sonnet','claude-3-haiku','claude-instant-v1','claude-v2:1', 'claude-v2'] 
        model=st.selectbox('**Model**', models,)
        params={"model":model}
        label_per_page=st.selectbox('**Label Per Page**', [True,False],)
        if "claude-3" in model:
            process_images=st.selectbox("**Document Processor**",["Textract","Claude3"],key="processor")
        else:
            process_images="Textract"
        labels = st.file_uploader("Upload manifest file", type=["txt"], accept_multiple_files=False)                      
        uploaded_files = st.file_uploader("Upload PDF or image files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
        params={"model":model,  "file_item":uploaded_files, "processor":process_images,"label_per_page":label_per_page, "possible_labels":labels}
        return params

def main():
    params=app_sidebar()
    page_summary(params)   
    
if __name__ == '__main__':
    main()   
