import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import shutil
import json
import pandas as pd
import torch.nn.functional as F
import monai.transforms as mtf
import re
import plotly.graph_objects as go
from monai.transforms import NormalizeIntensityd
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta
# import monai
import torchio as tio
from torchio import ScalarImage, Subject

# from ..utils.utils import mask2box
# from .dataset_info import dataset_info
import torchvision.transforms as transforms
from PIL import Image

B_INST, E_INST = "[INST]", "[/INST]"
IGNORE_INDEX = -100
PATCH_SIZE = 8
PATCH_SIZE_C = 10
PATCH_SIZE_SD = 5
IMAGE_SIZE = 320
CROP_SIZE = 192
FRAME_INTERVAL = 3

def _load_excel_columns(excel_path):
    if not excel_path or not os.path.exists(excel_path):
        return {}
    df = pd.read_excel(excel_path)
    return {column: df[column].tolist() for column in df.columns}

KM_excel_path = os.environ.get("KM_EXCEL_PATH", "data/NEW2014.9-2024.12.15CMR_concept_add_add_step10_Question_oc_R1_T.xlsx")
SCS_excel_path = os.environ.get("SCS_EXCEL_PATH", "data/SCS_CMR_dcm_step8_cleaned_T.xlsx")
CD_excel_path = os.environ.get("CD_EXCEL_PATH", "data/CD_515_step8_Question_oc_V3_merged_result3_T.xlsx")
YA_excel_path = os.environ.get("YA_EXCEL_PATH", "data/YA690_CMR3_S1.xlsx")

G_KM_columns_as_lists = _load_excel_columns(KM_excel_path)
G_SCS_columns_as_lists = _load_excel_columns(SCS_excel_path)
G_CD_columns_as_lists = _load_excel_columns(CD_excel_path)
G_YA_columns_as_lists = _load_excel_columns(YA_excel_path)
from .balance_loss import calculate_cardiac_weights
textsum_prompt = """-Goal-
Given the input text, please summarize the key findings and provide a differential diagnosis for each finding, for example, identify any enlargements, abnormalities, or reduced functionalities in the heart structures and report their potential causes. 
Additionally, provide any recommendations for further testing or evaluations based on the findings. 
Your response should be concise and focused on the most important information for decision-making.

######################
-Real Data-
######################
Text: 
{input_text}
######################
Output:
"""
caption_prompt =[
    # "Can you provide a radiology report for this medical image?",
    # "Describe the medical image you see.",
    # "What is depicted in this picture?",
    # "Please report this medical scan.",
    # "What is the medical significance of this image?",
    # "What can you infer from this picture?",
    # "Can you provide a quick summary of this image?",
    # "Describe this medical scan.",
    # "Please write a radiology report for this image.",
    # "Can you summarize the images presented?",
    # "Please generate a radiology report for this scan.",
    # "Describe the regions of interest in this scan.",
    "Please provide a caption for this medical image.",
    # "Can you provide a brief summary of this radiograph?",
    # "Describe the structures involved in this medical image.",
    # "What are the findings presented in this medical scan?",
    # "Please write a radiology report for this scan.",
    # "Can you provide a description of this medical scan?",
    # "Please caption this medical scan.",
    # "Can you provide a report summary for this medical scan?"
]

report_prompt =[
    "Can you provide a structured report for this CMR scan?",
    "请写一份结构化的CMR报告"
]

classification_prompt = [
    "What type of cardiovascular disease is shown in this image?",
    "Classify the heart condition visible in this scan.",
    "Identify the cardiovascular disorder depicted in this image.",
    "What is the primary cardiovascular diagnosis for this scan?",
    "Can you categorize the heart disease in this picture?",
    "What cardiovascular pathology is present in this image?",
    "Determine the type of cardiovascular disease in this scan.",
    "What heart condition is illustrated in this medical image?",
    "Classify the cardiovascular anomaly in this image.",
    "What is the specific cardiovascular disease in this scan?",
    "Identify the type of heart disease from this image.",
    "What cardiovascular issue is shown in this picture?",
    "Can you classify the heart disorder in this scan?",
    "What is the cardiovascular diagnosis based on this image?",
    "Determine the cardiovascular condition in this medical scan.",
    "What kind of heart disease is visible in this image?",
    "Classify the cardiovascular abnormality in this scan.",
    "What is the main cardiovascular finding in this image?",
    "Identify the cardiovascular disease type in this picture.",
    "What cardiovascular disorder is depicted in this scan?",
    "Can you determine the heart disease category from this image?",
    "What is the classification of the cardiovascular disease in this scan?",
    "What type of heart pathology is shown in this image?",
    "Classify the cardiovascular condition in this medical image.",
    "What cardiovascular disease is evident in this scan?",
    "What is the cardiovascular disease classification for this scan?",
    "Can you identify the type of cardiovascular disorder in this picture?",
    "What heart disease is represented in this medical image?",
    "Classify the cardiovascular issue in this scan."
]
question_open_prompt = [
"Please provide brief answers to the following questions based on the imaging studies and patient information.",
 "what is the answer to question",
"请根据影像和患者信息简要回答下述问题",
'请结合影像资料及病历信息简要回答以下问题',
'根据影像学检查结果与患者临床资料，请简要解答以下问题',
'基于患者影像报告和相关病史，请简明回复以下疑问',
'请简要作答',
'请根据影像回答下列问题（简明扼要）',
'请给出以下问题的简要答案',
'请综合分析影像检查，简要回答以下提问',
'需要您结合影像图像和患者基本信息简要回答这些问题',
'基于现有影像学证据和患者临床背景，请做出简明解答',
'请根据医学影像显示内容及相关资料，简要回复下述疑问',
]

question_close_prompt = [
"Select the single correct answer from given options. Respond format: 'Answer: [option letter]'",
"从给定选项中选出唯一正确答案。回答格式为：'Answer：选项字母",
]

additional_classification_prompt = '''-Goal-
Given images and the class list, please choose classes of the image from the class list.
The response format must be 'list', the list can include more than one class. 
The response must only include the given classes, separated by commas. Do not include any additional words or classes outside the given list.

Example 1: If the image is normal, just respond with ['Normal']. 
Example 2: If there exists 'Myocardial Infarction' and 'Hypertensive Heart Disease' simultaneously, output ['Myocardial Infarction','Hypertensive Heart Disease'].

- class list: ['Myocardial Infarction','Hypertensive Heart Disease','Dilated Cardiomyopathy','Hypertrophic Cardiomyopathy','Myocarditis','Normal','Noncompaction Cardiomyopathy']

######################
Output:
'''
additional_classification_prompt_larry = 'Please choose Heart Disease of this image.'

CLASSES = ['Myocardial Infarction',
           'Hypertensive Heart Disease',
           'Dilated Cardiomyopathy',
           'Hypertrophic Cardiomyopathy',
           'Myocarditis',
           'Normal',
           'Noncompaction Cardiomyopathy']

CLASSES_EN = [
    'Dilated Cardiomyopathy',
    'Hypertrophic Cardiomyopathy',
    'Myocarditis',
    'Cardiac Amyloidosis',
    'Myocardial Infarction',
    'Hypertensive Heart Disease',
    'Normal'
]
CLASSES_CN = [
    '扩张型心肌病',
    '肥厚型心肌病',
    '心肌炎',
    '心肌淀粉样变'
    '心肌梗死',
    '高血压心脏病',
    '正常',
]
CLASSES_EN_2 = [
    'Dilated Cardiomyopathy',          # 扩张型心肌病
    'Hypertrophic Cardiomyopathy',     # 肥厚型心肌病
    'Myocarditis',                     # 心肌炎
    'Myocardial Infarction',           # 心肌梗死
    'Hypertensive Heart Disease',      # 高血压心脏病
    'Normal',                          # 正常
    'Left Ventricular Noncompaction',  # 心肌致密化不全
    'Conduction System Disease'   
    # 'Cardiac Amyloidosis',# 传导系统疾病
]
CLASSES_CN_2 = [
    '扩张型心肌病',
    '肥厚型心肌病',
    '心肌炎',
    '心肌梗死',
    '高血压心脏病',
    '正常',
    '心肌致密化不全',
'传导系统疾病'
]
CLASSES_CN_3 = [
    '扩张型心肌病',
'肥厚型心肌病',
'正常',
    '心肌梗死',
'高血压心脏病',
    '心肌炎',
]
CLASSES_EN_3 = [
    'Dilated Cardiomyopathy',          # 扩张型心肌病
    'Hypertrophic Cardiomyopathy',     # 肥厚型心肌病
    'Normal',  # 正常
    'Myocardial Infarction',           # 心肌梗死
    'Hypertensive Heart Disease',      # 高血压心脏病
    'Myocarditis',  # 心肌炎
]
CLASSES_CN_4 = [
    '扩张型心肌病',
    '肥厚型心肌病',
    '心肌炎',
    '心肌梗死',
    '高血压心脏病',
    '正常',
    '心肌淀粉样变',
'传导系统疾病',
   '心肌致密化不全',
    '应激性心肌病',
   '致心律失常性',
   '心包炎',
   '肺动脉高压',
    '心脏瓣膜病',
   '先天性心脏病',
    '肿瘤相关',
  '产褥期相关心肌病',
  '系统性疾病相关心脏病',
    '卒中'
]

CLASSES_EN_4 = [
    ('Dilated Cardiomyopathy', 'DCM'),
    ('Hypertrophic Cardiomyopathy', 'HCM'),
    ('Myocarditis', 'M'),   #Myocarditis Associated with Non-autoimmune Rheumatic Diseases
    ('Myocardial Infarction', 'MI'),
    ('Hypertensive Heart Disease', 'HHD'),
    ('Normal', 'N'),
    ('Cardiac Amyloidosis', 'CA'),
    ('Conduction System Disease', 'CSD'),
    ('Noncompaction Cardiomyopathy', 'NCM'),  # or LVNC (Left Ventricular Noncompaction)
    ('Takotsubo Cardiomyopathy', 'TCM'),  # or TTC (Takotsubo Cardiomyopathy)
    ('Arrhythmogenic Cardiomyopathy', 'ACM'),  # or ARVC (Arrhythmogenic Right Ventricular Cardiomyopathy)
    ('Pericarditis', 'P'),
    ('Pulmonary Hypertension', 'PH'),
    ('Valvular Heart Disease', 'VHD'),
    ('Congenital Heart Disease', 'CHD'),
    ('Tumor-related Heart Disease', 'TRHD'),
    ('Peripartum Cardiomyopathy', 'PCM'),  # or PPCM
    ('Systemic Disease-related Heart Disease', 'SDRHD'), #Myocarditis Associated with Autoimmune Rheumatic Diseases
    ('Stroke', 'S')
]


# '心肌淀粉样变',
   # '心肌致密化不全',
   #  '应激性心肌病',
  # '限制性心肌病',
  #  '致心律失常性',
   # '心包炎',
   # '肺动脉高压',
   #  '心脏瓣膜病',
   #  '传导系统疾病',
   # '先天性心脏病',
   #  '肿瘤相关',
  # '产褥期相关心肌病',
  # '系统性疾病相关心脏病',



name_mapping = {
    # 心腔尺寸
    'LVEDD': 'LV End-Diastolic Diameter (LVEDD)',
    '左心房前后径': 'Left Atrial Anteroposterior Diameter',
    'RVEDD': 'RV End-Diastolic Diameter (RVEDD)',
    '右心房内径': 'Right Atrial Diameter',

    # 心肌特征
    '左心室壁厚度': 'LV Wall Thickness',
    # '最厚处厚度': 'Maximal Wall Thickness',
    # '变薄区域': 'Regional Wall Thinning',

    # 心脏功能
    'LV室壁运动': 'LV Wall Motion',
    # '运动异常范围': 'Wall Motion Abnormality Extent',
    # '异常部位': 'Location of Abnormalities',
    '室壁瘤': 'Ventricular Aneurysm',

    # 射血分数和容积
    'LVEF': 'Left Ventricular Ejection Fraction (LVEF)',
    'LVEDV': 'LV End-Diastolic Volume (LVEDV)',
    'LVESV': 'LV End-Systolic Volume (LVESV)',
    'RVEF': 'Right Ventricular Ejection Fraction (RVEF)',
    'RVEDV': 'RV End-Diastolic Volume (RVEDV)',
    'RVESV': 'RV End-Systolic Volume (RVESV)',

    # 瓣膜功能
    '二尖瓣返流': 'Mitral Regurgitation',
    '三尖瓣返流': 'Tricuspid Regurgitation',
    '主动脉瓣返流': 'Aortic Regurgitation',

    # 延迟强化
    '左心室LGE': 'LV Late Gadolinium Enhancement (LGE)',
    # 'LGE部位': 'LGE Location',
    # 'LGE分布': 'LGE Distribution Pattern',
    # 'LGE透壁程度': 'LGE Transmural Extent',
    '右心室LGE': 'RV Late Gadolinium Enhancement (RV-LGE)',
    # 'RV-LGE部位': 'RV-LGE Location',

    # 其他发现
    '心包强化': 'Pericardial Enhancement',
    '心肌脂肪浸润': 'Myocardial Fatty Infiltration',
    # '脂肪浸润部位': 'Fatty Infiltration Location',
    '血栓': 'Intracardiac Thrombus',
    # '血栓部位': 'Thrombus Location',
    # '血栓大小': 'Thrombus Size',
    '心包积液': 'Pericardial Effusion',
    '胸腔积液': 'Pleural Effusion'
}
import matplotlib.pyplot as plt

def extract_content(data_str, name_id = 0):
    target_section_pattern_list = [re.compile(r'1\.\s*心脏结构\"?\s*:\s*"((?:[^"]|\n)*?)"\s*',flags=re.IGNORECASE),
                                   re.compile(r'2\.\s*心脏运动及功能\"?\s*:\s*"((?:[^"]|\n)*?)"\s*', flags=re.IGNORECASE),
                                   re.compile(r'3\.\s*(?:延迟强化)LGE\"?\s*:\s*"((?:[^"]|\n)*?)"\s*', flags=re.IGNORECASE),
                                   re.compile(r'4\.\s*其他影像所见\"?\s*:\s*"((?:[^"]|\n)*?)"\s*', flags=re.IGNORECASE),
                                   re.compile(r'2\.\s*专业医学问题\"?\s*:\s*"((?:[^"]|\n)*?)"\s*', flags=re.IGNORECASE),
                                   re.compile(r'3\.\s*非医学问题\"?\s*:\s*"((?:[^"]|\n)*?)"\s*', flags=re.IGNORECASE),
        ]
    match = target_section_pattern_list[name_id].search(data_str)
    if not match:
        print("未找到")
        return None
        # raise ValueError("未找到'1. 心脏结构'部分")

    content = match.group(1)

    # 第二步：提取问答对
    qa_pattern = re.compile(
        r'\*Question-\d+:\s*([^？]+[？])\s*Answer:\s*([^。*]+)',
        flags=re.DOTALL
    )

    result = []
    for question, answer in qa_pattern.findall(content):
        result.append({
            "question": question.strip(),
            "answer": answer.strip().rstrip('。').rstrip('，')  # 去除结尾句号
        })

    # 打印结果
    # print("提取到 {} 个问答对:".format(len(result)))
    return result
    # for i, qa in enumerate(result, 1):
    #     print(f"{i}. {qa['question']}")
    #     print(f"   → {qa['answer']}\n")


def most_common_number(lst):
    frequency = {}
    for num in lst:
        frequency[num] = frequency.get(num, 0) + 1
    # 返回出现次数最多的数字
    return max(frequency, key=frequency.get)

def extract_content_close(data_str, name_id = 0):
    target_section_pattern_list = [re.compile(r'1\.\s*心脏结构\"?\s*:\s*"((?:[^"]|\n)*?)"\s*',flags=re.IGNORECASE),
                                   re.compile(r'2\.\s*心脏运动及功能\"?\s*:\s*"((?:[^"]|\n)*?)"\s*', flags=re.IGNORECASE),
                                   re.compile(r'3\.\s*(?:延迟强化)LGE\"?\s*:\s*"((?:[^"]|\n)*?)"\s*', flags=re.IGNORECASE),
                                   re.compile(r'4\.\s*其他影像所见\"?\s*:\s*"((?:[^"]|\n)*?)"\s*', flags=re.IGNORECASE),
        ]
    match = target_section_pattern_list[name_id].search(data_str)
    if not match:
        print("未找到")
        return None
        # raise ValueError("未找到'1. 心脏结构'部分")

    content = match.group(1)

    blocks = re.findall(r'\*(.*?)\*', content, flags=re.DOTALL)

    questions = []

    for block in blocks:

        question_match = re.search(r'Question-\d+:\s*(.*?)\s*Choice:', block, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""

        # 提取选项部分
        options_match = re.search(r'Choice:\s*(.*?)\s*Answer:', block, re.DOTALL)
        options_str = options_match.group(1).strip() if options_match else ""

        # 分割选项
        options = {}
        option_letters = []  # 存储原始选项顺序
        if options_str:
            option_parts = re.split(r'\s+(?=[A-D]\.)', options_str)
            for part in option_parts:
                match = re.match(r'([A-D])\.\s*(.*)', part)
                if match:
                    letter = match.group(1)
                    options[letter] = match.group(2).strip()
                    option_letters.append(letter)  # 记录原始顺序

            random.shuffle(option_letters)  # 核心改动点

        # 提取答案
        answer_match = re.search(r'Answer:\s*([A-D])\.\s*(.*)', block)
        answer_letter = answer_match.group(1) if answer_match else ""
        answer_text = answer_match.group(2).strip() if answer_match else ""

        # 存储结果
        shuffled_options=[]
        randanswer_letter = ''
        for abcd, abcd_2 in zip(option_letters, ['A','B','C','D'][:len(option_letters)]):
            abcd_text = options[abcd]
            shuffled_options.append((abcd_2, abcd_text))
            if abcd == answer_letter:
                randanswer_letter = abcd_2


        questions.append({
        'question': question,
        'options': options,
        'option_letters': option_letters,  # 新增打乱后的顺序
        'answer_letter': answer_letter,
        'answer_text': answer_text,
        'shuffled_options': shuffled_options,
        'randanswer_letter': randanswer_letter
        })

    # 打印结果
    # print("提取到 {} 个问答对:".format(len(result)))
    return questions
    # for i, qa in enumerate(result, 1):
    #     print(f"{i}. {qa['question']}")
    #     print(f"   → {qa['answer']}\n")


def show_mask(mask, ax, i, random_color=True, borders = True):
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    color = [np.array([0 / 255, 0 / 255, 255 / 255, 0.5]),
             np.array([0 / 255, 255 / 255, 0 / 255, 0.5]),
             np.array([255 / 255, 0 / 255, 0 / 255, 0.5]),
             ][i]
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_masks(image, masks, borders=False,):
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    # for i in range(0,3):
    #     mask = masks[i]
    #     show_mask(mask, plt.gca(), i, borders=borders)
    show_mask(masks, plt.gca(), 0, borders=borders)
    # plt.title(text, fontsize=8)
    plt.axis('off')
    plt.show
    # plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
    # plt.close()

def preprocess_image(
    image_tensor: torch.Tensor,
    patch_size=PATCH_SIZE
) -> torch.Tensor:
    # Reshape the image to get the patches
    # shape changes: (C=3, H, W)
    # -> (C, N_H_PATCHES, W, PATCH_H)
    # -> (C, N_H_PATCHES, N_W_PATCHES, PATCH_H, PATCH_W)
    # import pdb;pdb.set_trace()
    patches = image_tensor.unfold(1, patch_size, patch_size)\
        .unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous() # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
    return patches

def preprocess_image3D(
    image_tensor: torch.Tensor,
    patch_size=PATCH_SIZE
) -> torch.Tensor:
    channels, depth, height, width = image_tensor.size()
    pad_d = (PATCH_SIZE_C - depth % PATCH_SIZE_C) % PATCH_SIZE_C
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    # 对张量进行 padding
    padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h, 0, pad_d))
    # 使用 unfold 操作切分张量
    patches = padded_tensor.unfold(1, PATCH_SIZE_C, PATCH_SIZE_C) \
        .unfold(2, patch_size, patch_size) \
        .unfold(3, patch_size, patch_size)

    # 调整张量形状
    patches = patches.permute(1, 2, 3, 0, 4, 5, 6).contiguous()
    return patches


def preprocess_image3D_vst(
        image_tensor: torch.Tensor,
        patch_size: int = 4,  # 空间维度patch大小 (h, w)
        patch_size_c: int = 4,  # 深度维度patch大小 (d)
        window_size = (2, 7, 7)  # Swin窗口划分粒度
):
    """
    改进后的3D图像预处理：
    1. 动态计算padding确保各维度满足：
       - 原始patch划分
       - Swin窗口机制的分层需求
    2. 返回处理后的patch张量及token形状

    参数说明：
    window_size - 表示每个维度最少需要划分的窗口数（例如(2,2,2)表示每个维度至少分成2个窗口）
    """
    channels, depth, height, width = image_tensor.size()

    # 计算满足双重要求的padding（既要整除patch_size，又要满足窗口划分）
    def calc_pad(dim, patch, win):
        # 总长度需要满足：total_length = patch_size * window_size * n
        required = patch * win
        return (required - dim % required) % required

    pad_d = calc_pad(depth, patch_size_c, window_size[0])
    pad_h = calc_pad(height, patch_size, window_size[1])
    pad_w = calc_pad(width, patch_size, window_size[2])

    # 执行padding (PyTorch的pad顺序是反的：width, height, depth)
    padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h, 0, pad_d))

    # 三维unfold分解
    patches = padded_tensor.unfold(1, patch_size_c, patch_size_c) \
        .unfold(2, patch_size, patch_size) \
        .unfold(3, patch_size, patch_size)

    # 维度重排 [batch]->[D_patches, H_patches, W_patches, C, patch_d, patch_h, patch_w]
    patches = patches.permute(1, 2, 3, 0, 4, 5, 6).contiguous()

    # 计算token形状 (用于后续的window划分)
    token_shape = (
        (depth + pad_d) // patch_size_c,
        (height + pad_h) // patch_size,
        (width + pad_w) // patch_size
    )

    return patches, token_shape



def preprocess_imageSD(
    image_tensor: torch.Tensor,
    patch_size=PATCH_SIZE
) -> torch.Tensor:
    # import pdb;pdb.set_trace()
    channels, depth, height, width = image_tensor.size()
    pad_d = (PATCH_SIZE_SD - depth % PATCH_SIZE_SD) % PATCH_SIZE_SD
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size

    # 对张量进行 padding
    padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h, 0, pad_d))
    # 使用 unfold 操作切分张量
    patches = padded_tensor.unfold(1, PATCH_SIZE_SD, PATCH_SIZE_SD) \
        .unfold(2, patch_size, patch_size) \
        .unfold(3, patch_size, patch_size)

    # 调整张量形状
    patches = patches.permute(1, 2, 3, 0, 4, 5, 6).contiguous()
    # d_num, h_num, w_num, channels, depth, patch_h_size, patch_w_size)
    # patches = patches.contiguous().view(-1, channels, PATCH_SIZE_SD, patch_size, patch_size)
    return patches

def preprocess_imagefilm(
    image_tensor: torch.Tensor,
    patch_size=PATCH_SIZE
) -> torch.Tensor:
    # import pdb;pdb.set_trace()
    channels, depth, height, width = image_tensor.size()
    pad_d = (PATCH_SIZE_C - depth % PATCH_SIZE_C) % PATCH_SIZE_C
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size

    # 对张量进行 padding
    padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h, 0, pad_d))
    # 使用 unfold 操作切分张量
    patches = padded_tensor.unfold(1, PATCH_SIZE_C, PATCH_SIZE_C) \
        .unfold(2, patch_size, patch_size) \
        .unfold(3, patch_size, patch_size)

    # 调整张量形状
    patches = patches.permute(2, 3, 1, 0, 4, 5, 6).contiguous()
    #  h_num, w_num, d_num, channels, depth, patch_h_size, patch_w_size)
    # patches = patches.contiguous().view(-1, channels, PATCH_SIZE_C, patch_size, patch_size)
    return patches

def get_transform():
    preprocess_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                std=[0.229, 0.224, 0.225])   # standard deviation for pre-trained models on ImageNet
        ])
    return preprocess_transform

def get_transform2():
    preprocess_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                std=[0.229, 0.224, 0.225])   # standard deviation for pre-trained models on ImageNet
        ])
    return preprocess_transform





class QADataset_CMR(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_position_embeddings = 2048
        with open(args.qa_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        set_track_meta(False)



    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, Question, Answer, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        question = Question
        answer = Answer

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        answer = answer + end_token
        _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend(cur_tokens)
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels


    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                Question = data["Question"]
                Answer = data["Answer"]
                tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_img(
                    Question, Answer, self.tokenizer)
                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patches": vision_patches,
                    "vision_patch_indices": vision_patch_indices,
                    "labels": labels
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class TextSumDataset_CMR(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_position_embeddings = 2048
        with open(args.all_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file

        set_track_meta(False)

    def __len__(self):
        return len(self.data_list[self.mode])

    def prepare_inputs_txt(self, question, answer, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        answer = answer + end_token
        _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend(cur_tokens)
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        # print(len(tokens))
        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels


    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[self.mode][idx]
                output_text_list = []
                # excel 诊断意见	检查所见  id
                if data['Text'].get('text_Findings_4CH') is None or str(data['Text']['text_Findings_4CH']) == 'nan' or type(data['Text']['text_Findings_4CH']) in [int,float]:
                    output_text_list.append('Cardiac Shape: No finding is observed.')
                else:
                    output_text_list.append('Cardiac Shape: '+data['Text']['text_Findings_4CH'])
                
                if data['Text'].get('text_Findings_SAX') is None or str(data['Text']['text_Findings_SAX']) == 'nan' or type(data['Text']['text_Findings_SAX']) in [int,float]:
                    output_text_list.append('Cine: No finding is observed.')
                else:
                    output_text_list.append('Cine: '+data['Text']['text_Findings_SAX'])
                
                if data['Text'].get('text_Findings_LGE') is None or str(data['Text']['text_Findings_LGE']) == 'nan' or type(data['Text']['text_Findings_LGE']) in [int,float]:
                    output_text_list.append('LGE: No finding is observed.')
                else:
                    output_text_list.append('LGE: '+data['Text']['text_Findings_LGE'])
                
                
                Question = textsum_prompt.format(input_text=' '.join(output_text_list))
                # print(Question)
                Answer = data['Text'].get('text_Impression')


                tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_txt(
                    Question, Answer, self.tokenizer)
                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patch_indices": vision_patch_indices,
                    "labels": labels
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class ReadDataset_CMR(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_position_embeddings = 2048
        with open(args.qa_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        set_track_meta(False)



    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, Question, Answer, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        question = Question
        answer = Answer

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        answer = answer + end_token
        _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend(cur_tokens)
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels


    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                Question = data["Question"]
                Answer = data["Answer"]
                tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_img(
                    Question, Answer, self.tokenizer)
                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patches": vision_patches,
                    "vision_patch_indices": vision_patch_indices,
                    "labels": labels
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class TextDataset_CMR(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_position_embeddings = 2048
        with open(args.qa_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file

        set_track_meta(False)



    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, Question, Answer, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []
        sentences = Question.split('.')
        if len(sentences) > 1:
            sentences = [item for item in sentences if len(item) >= 10]
            id = random.randint(1, len(sentences))
            question_sentences = [sentence.strip() for sentence in sentences[:id]]
            answer_sentences = [sentence.strip() for sentence in sentences[id-1:]]
            question = '. '.join(question_sentences)
            answer = '. '.join(answer_sentences)
        else:
            b = min(50, len(Question))
            start = random.randint(b, len(Question))
            question = Question[:start]
            answer = Answer

        # answer = Answer

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        answer = answer + end_token
        _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend(cur_tokens)
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels


    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                Question = data["text_paragraph"]
                Answer = data["text_paragraph"]
                tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_img(
                    Question, Answer, self.tokenizer)
                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patches": vision_patches,
                    "vision_patch_indices": vision_patch_indices,
                    "labels": labels
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class LocationDataset_CMR_3D(Dataset):
    def __init__(self, args, tokenizer, mode="train", random_slize = False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.max_position_embeddings = 4096
        with open(args.location_data_path3D, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])

        val_transform =  transforms.Compose([
            transforms.ToTensor(),  #Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])
        # train_transform = mtf.Compose(
        #     [
        #         mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        #         mtf.RandFlip(prob=0.10, spatial_axis=0),
        #         mtf.RandFlip(prob=0.10, spatial_axis=1),
        #         mtf.RandFlip(prob=0.10, spatial_axis=2),
        #         mtf.RandScaleIntensity(factors=0.1, prob=0.5),
        #         mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
        #
        #         mtf.ToTensor(dtype=torch.float),
        #     ]
        # )
        #
        # val_transform = mtf.Compose(
        #     [
        #         mtf.ToTensor(dtype=torch.float),
        #     ]
        # )

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        set_track_meta(False)



    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        patches = images
        n_deeps, n_rows, n_cols = patches.shape[:3]
        n_patches = n_deeps * n_rows * n_cols
        patches = patches.view(n_patches, -1)

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):

                    if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vframe_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    if row_idx != 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(f"<vpatch>")
                    cur_patch_indices.append(len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)

        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)
        # print(f"cur_tokens: {cur_tokens}")
        # print(f"cur_attention_mask: {cur_attention_mask}")
        # print(f"cur_patch_indices: {cur_patch_indices}")
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))

        caption_prompt =[
            "Can you identify the cardiac segmentation (Basal Segments, Mid-cavity Segments, Apical Segments) represented in this medical image?",
            "Which cardiac segmentation does this imaging scan correspond to?",
            "What cardiac segmentation (Basal Segments, Mid-cavity Segments, Apical Segments) is depicted in this medical imaging?",
            "Could you specify which segmentation of the heart this image illustrates?",
            "How can we determine the cardiac segmentation shown in this scan?",
            "Is it possible to tell which cardiac segmentation (Basal Segments, Mid-cavity Segments, Apical Segments) is featured in this image?",
            "Can this medical image be categorized into a specific cardiac segment?",
            "What segmentation (Basal Segments, Mid-cavity Segments, Apical Segments) of the heart can be identified in this imaging study?"
            ]

        question = random.choice(caption_prompt)
        answer = inputs

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        answer = answer + end_token
        _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend(cur_tokens)
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        # for idx, i in enumerate(input_text):
        #     i = i['value']
        #     if idx % 2 == 0:
        #         if idx == 0:
        #             i = i.replace("<image>\n", '').replace("\n<image>", '')
        #             c_new = tokenizer.bos_token + f"{B_INST} {i.strip()} {E_INST}"
        #         else:
        #             c_new = f"{B_INST} {i.strip()} {E_INST}"
        #         _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        #         cur_tokens = _tokenized["input_ids"].squeeze(0)
        #         cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        #         tokens.extend(cur_tokens)
        #         labels.extend([-100] * len(cur_tokens))
        #         attention_masks.extend(cur_attention_mask)
        #         vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        #     else:
        #         i = i + end_token
        #         _tokenized = tokenizer(i, return_tensors="pt", add_special_tokens=False)
        #         cur_tokens = _tokenized["input_ids"].squeeze(0)
        #         cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        #         tokens.extend(cur_tokens)
        #         labels.extend(cur_tokens)
        #         attention_masks.extend(cur_attention_mask)
        #         vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels

    def prepare_inputs_img_test(self, images, inputs, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        patches = images
        n_deeps, n_rows, n_cols = patches.shape[:3]
        n_patches = n_deeps * n_rows * n_cols
        patches = patches.view(n_patches, -1)

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):

                    if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vframe_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    if row_idx != 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(f"<vpatch>")
                    cur_patch_indices.append(
                        len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)

        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)
        # print(f"cur_tokens: {cur_tokens}")
        # print(f"cur_attention_mask: {cur_attention_mask}")
        # print(f"cur_patch_indices: {cur_patch_indices}")
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))

        caption_prompt = [
            "Can you identify the cardiac segmentation (Basal Segments, Mid-cavity Segments, Apical Segments) represented in this medical image?",
            "Which cardiac segmentation does this imaging scan correspond to?",
            "What cardiac segmentation (Basal Segments, Mid-cavity Segments, Apical Segments) is depicted in this medical imaging?",
            "Could you specify which segmentation of the heart this image illustrates?",
            "How can we determine the cardiac segmentation shown in this scan?",
            "Is it possible to tell which cardiac segmentation (Basal Segments, Mid-cavity Segments, Apical Segments) is featured in this image?",
            "Can this medical image be categorized into a specific cardiac segment?",
            "What segmentation (Basal Segments, Mid-cavity Segments, Apical Segments) of the heart can be identified in this imaging study?"
        ]

        question = random.choice(caption_prompt)
        answer = inputs

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        # print(inputs)
        # answer = answer + end_token
        # _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        # cur_tokens = _tokenized["input_ids"].squeeze(0)
        # cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        # tokens.extend(cur_tokens)
        # labels.extend(cur_tokens)
        # attention_masks.extend(cur_attention_mask)
        # vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        # for idx, i in enumerate(input_text):
        #     i = i['value']
        #     if idx % 2 == 0:
        #         if idx == 0:
        #             i = i.replace("<image>\n", '').replace("\n<image>", '')
        #             c_new = tokenizer.bos_token + f"{B_INST} {i.strip()} {E_INST}"
        #         else:
        #             c_new = f"{B_INST} {i.strip()} {E_INST}"
        #         _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        #         cur_tokens = _tokenized["input_ids"].squeeze(0)
        #         cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        #         tokens.extend(cur_tokens)
        #         labels.extend([-100] * len(cur_tokens))
        #         attention_masks.extend(cur_attention_mask)
        #         vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        #     else:
        #         i = i + end_token
        #         _tokenized = tokenizer(i, return_tensors="pt", add_special_tokens=False)
        #         cur_tokens = _tokenized["input_ids"].squeeze(0)
        #         cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        #         tokens.extend(cur_tokens)
        #         labels.extend(cur_tokens)
        #         attention_masks.extend(cur_attention_mask)
        #         vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels, question


    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image_path"]
                image_abs_path = image_path
                image_list = []
                for item_image in image_path:
                    image = Image.open(str(item_image))
                    image = self.transform(image)
                    image_list.append(image[np.newaxis, ...])
                # image = Image.open(str(image_abs_path)) # nomalized 0-1, C,D,H,W
                # image = np.load(image_abs_path)[np.newaxis, ...]  # nomalized
                # width, height = image.size
                # image = self.transform(image)
                # image = self.transform(image)
                images = torch.cat(image_list, dim=0).transpose(0, 1)

                
                current_frames = images.size(1)
                target_frames = min(30,current_frames)
                if current_frames > target_frames:
                    # 随机选择起始帧
                    if self.mode == 'train':
                        start_frame = torch.randint(0, current_frames - target_frames + 1, (1,)).item()
                        images = images[:, start_frame:start_frame + target_frames, :, :]
                    else:
                        images = images[:, 0:30, :, :]

                images = images[:,::FRAME_INTERVAL,:,:]


                img_patches = preprocess_image3D(images)

                text = data["text"]


                if self.mode == 'train':
                    tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_img(
                        img_patches, text, self.tokenizer)
                    ret = {
                        "input_ids": tokens,
                        "attention_mask": attention_masks,
                        "sax_vision_patches": vision_patches,
                        "vision_patch_indices": vision_patch_indices,
                        "labels": labels
                    }
                else:
                    tokens, attention_masks, vision_patches, vision_patch_indices, labels, question = self.prepare_inputs_img_text(
                        img_patches, text, self.tokenizer)
                    ret = {
                        "input_ids": tokens,
                        "attention_mask": attention_masks,
                        "sax_vision_patches": vision_patches,
                        "vision_patch_indices": vision_patch_indices,
                        "labels": labels,
                        "text": text,
                        "image_path": image_path,
                        "question": question
                    }

                # input_id = text_tensor["input_ids"][0]
                # attention_mask = text_tensor["attention_mask"][0]
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class SD_Dataset_CMR_3D(Dataset):
    def __init__(self, args, tokenizer, mode="train", random_slize = False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.max_position_embeddings = 4096
        with open(args.sd_data_path3D, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])

        val_transform =  transforms.Compose([
            transforms.ToTensor(),  #Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])


        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        set_track_meta(False)



    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        patches = images
        n_deeps, n_rows, n_cols = patches.shape[:3]
        n_patches = n_deeps * n_rows * n_cols
        patches = patches.view(n_patches, -1)

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):

                    if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vframe_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    if row_idx != 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(f"<vpatch>")
                    cur_patch_indices.append(len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)

        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)
        # print(f"cur_tokens: {cur_tokens}")
        # print(f"cur_attention_mask: {cur_attention_mask}")
        # print(f"cur_patch_indices: {cur_patch_indices}")
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))

        caption_prompt =[
            "What cardiac phase is shown in the given image?",
            "Can you determine the cardiac phase illustrated in this image?",
            "WWhich phase of the cardiac cycle does the image represent?",
            "Identify the cardiac phase visible in the image provided",
            "What stage of the heart cycle is depicted in the image?",
            "Could you specify the cardiac phase shown in the image?",
            "What part of the cardiac cycle is represented in this image?",
            "Can you recognize the cardiac phase displayed in this image?",
            "What cardiac cycle phase is being visualized in the image?",
            "Please identify the cardiac phase depicted in the image?"
            ]

        question = random.choice(caption_prompt)
        answer = inputs

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        answer = answer + end_token
        _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend(cur_tokens)
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels

    def prepare_inputs_img_text(self, images, inputs, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        patches = images
        n_deeps, n_rows, n_cols = patches.shape[:3]
        n_patches = n_deeps * n_rows * n_cols
        patches = patches.view(n_patches, -1)

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):

                    if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vframe_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    if row_idx != 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(f"<vpatch>")
                    cur_patch_indices.append(
                        len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)

        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)
        # print(f"cur_tokens: {cur_tokens}")
        # print(f"cur_attention_mask: {cur_attention_mask}")
        # print(f"cur_patch_indices: {cur_patch_indices}")
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))

        caption_prompt = [
            "What cardiac phase is shown in the given image?",
            "Can you determine the cardiac phase illustrated in this image?",
            "WWhich phase of the cardiac cycle does the image represent?",
            "Identify the cardiac phase visible in the image provided",
            "What stage of the heart cycle is depicted in the image?",
            "Could you specify the cardiac phase shown in the image?",
            "What part of the cardiac cycle is represented in this image?",
            "Can you recognize the cardiac phase displayed in this image?",
            "What cardiac cycle phase is being visualized in the image?",
            "Please identify the cardiac phase depicted in the image?"]

        question = random.choice(caption_prompt)
        answer = inputs

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels, question


    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image_path"]
                image_abs_path = image_path
                image_list = []
                for item_image in image_path:
                    image = Image.open(str(item_image))
                    image = self.transform(image)
                    image_list.append(image[np.newaxis, ...])
                # image = Image.open(str(image_abs_path)) # nomalized 0-1, C,D,H,W
                # image = np.load(image_abs_path)[np.newaxis, ...]  # nomalized
                # width, height = image.size
                # image = self.transform(image)
                # image = self.transform(image)
                images = torch.cat(image_list, dim=0).transpose(0, 1)
                
                img_patches = preprocess_imageSD(images)
                # print(img_patches.shape)

                text = data["text"]


                if self.mode == 'train':
                    tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_img(
                        img_patches, text, self.tokenizer)
                    ret = {
                        "input_ids": tokens,
                        "attention_mask": attention_masks,
                        "vision_patches": vision_patches,
                        "vision_patch_indices": vision_patch_indices,
                        "labels": labels
                    }
                else:
                    tokens, attention_masks, vision_patches, vision_patch_indices, labels, question = self.prepare_inputs_img_text(
                        img_patches, text, self.tokenizer)
                    ret = {
                        "input_ids": tokens,
                        "attention_mask": attention_masks,
                        "vision_patches": vision_patches,
                        "vision_patch_indices": vision_patch_indices,
                        "labels": labels,
                        "text": text,
                        "image_path": image_path,
                        "question": question
                    }

                # input_id = text_tensor["input_ids"][0]
                # attention_mask = text_tensor["attention_mask"][0]
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class LV_Dataset_CMR_3D(Dataset):
    def __init__(self, args, tokenizer, mode="train", random_slize = False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.max_position_embeddings = 4096
        with open(args.lv_data_path3D, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])

        val_transform =  transforms.Compose([
            transforms.ToTensor(),  #Convert the image to a PyTorch tensor
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        set_track_meta(False)



    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        patches = images
        n_deeps, n_rows, n_cols = patches.shape[:3]
        n_patches = n_deeps * n_rows * n_cols
        patches = patches.view(n_patches, -1)

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):

                    if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vframe_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    if row_idx != 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(f"<vpatch>")
                    cur_patch_indices.append(len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)

        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)
        # print(f"cur_tokens: {cur_tokens}")
        # print(f"cur_attention_mask: {cur_attention_mask}")
        # print(f"cur_patch_indices: {cur_patch_indices}")
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))

        caption_prompt =[
            'Can you determine the left ventricular volume shown in the image?',
            'What is the measured left ventricular volume in this image?',
            'Identify the left ventricular volume depicted in the image.',
            'How much is the left ventricular volume in the given image?',
            'What is the calculated left ventricular volume in this image?',
            'Can you specify the left ventricular volume seen in the image?',
            'What is the volume of the left ventricle as shown in the image?',
            'How large is the left ventricular volume in the provided image?',
            'What is the estimated left ventricular volume in this image?',
            'Could you determine the left ventricular volume depicted in the image?',
            ]

        question = random.choice(caption_prompt)
        answer = inputs

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        answer = answer + end_token
        _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend(cur_tokens)
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels

    def prepare_inputs_img_text(self, images, inputs, tokenizer):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        patches = images
        n_deeps, n_rows, n_cols = patches.shape[:3]
        n_patches = n_deeps * n_rows * n_cols
        patches = patches.view(n_patches, -1)

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):

                    if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vframe_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    if row_idx != 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(f"<vpatch>")
                    cur_patch_indices.append(
                        len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)

        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)
        # print(f"cur_tokens: {cur_tokens}")
        # print(f"cur_attention_mask: {cur_attention_mask}")
        # print(f"cur_patch_indices: {cur_patch_indices}")
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))

        caption_prompt = [
           ' Can you determine the left ventricular volume shown in the image?',
            'What is the measured left ventricular volume in this image?',
            'Identify the left ventricular volume depicted in the image.',
            'How much is the left ventricular volume in the given image?',
            'What is the calculated left ventricular volume in this image?',
            'Can you specify the left ventricular volume seen in the image?',
            'What is the volume of the left ventricle as shown in the image?',
            'How large is the left ventricular volume in the provided image?',
            'What is the estimated left ventricular volume in this image?',
            'Could you determine the left ventricular volume depicted in the image?',
        ]

        question = random.choice(caption_prompt)
        answer = inputs

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels, question


    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image_path"]
                image_abs_path = image_path
                image_list = []
                for item_image in image_path:
                    image = Image.open(str(item_image))
                    image = self.transform(image)
                    image_list.append(image[np.newaxis, ...])
                # image = Image.open(str(image_abs_path)) # nomalized 0-1, C,D,H,W
                # image = np.load(image_abs_path)[np.newaxis, ...]  # nomalized
                # width, height = image.size
                # image = self.transform(image)
                # image = self.transform(image)
                images = torch.cat(image_list, dim=0).transpose(0, 1)

                # if self.random_slize:
                #     target_frames_list = [15,20,25,30]
                #     target_frames = random.choice(target_frames_list)
                # else:
                #     target_frames = 25
                current_frames = images.size(1)
                if current_frames > 5:
                    numbers = [0, 1, 1, 2, 2, 3]
                    random_numbers = random.sample(numbers, 2)
                    images = images[:, random_numbers[0]:current_frames-random_numbers[-1], :, :]

                img_patches = preprocess_image3D(images) #

                text = data["text"]
                y_true_number = re.search(r"\d+\.?\d*", text)
                y_true_number = torch.tensor(float(y_true_number.group()))
                if self.mode == 'train':
                    tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_img(
                        img_patches, text, self.tokenizer)
                    ret = {
                        "input_ids": tokens,
                        "attention_mask": attention_masks,
                        "vision_patches": vision_patches,
                        "vision_patch_indices": vision_patch_indices,
                        "labels": labels,
                        'score_lv_button': True,
                        'score_lv_label': y_true_number
                    }
                else:
                    tokens, attention_masks, vision_patches, vision_patch_indices, labels, question = self.prepare_inputs_img_text(
                        img_patches, text, self.tokenizer)
                    ret = {
                        "input_ids": tokens,
                        "attention_mask": attention_masks,
                        "vision_patches": vision_patches,
                        "vision_patch_indices": vision_patch_indices,
                        "labels": labels,
                        "text": text,
                        "image_path": image_path,
                        "question": question,
                        'score_lv_button': True,
                        'score_lv_label': y_true_number
                    }

                # input_id = text_tensor["input_ids"][0]
                # attention_mask = text_tensor["attention_mask"][0]
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)




class SAXCineDataset_CMR_3DFilm_vst(Dataset):
    def __init__(self, args, data_root, tokenizer, json_file,
                 G_columns_as_lists, mode="train", random_slize=False, use_seg=False, use_numpy = False, use_det = False):
        self.args = args
        # self.data_root = args.data_root
        self.data_root = data_root['image']
        self.seg_Lv_root = data_root['LV_mask']
        self.seg_MYO_root = data_root['MYO_mask']
        self.det_root  = data_root['DET_mask']
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.use_seg = use_seg
        self.max_position_embeddings = 4096
        self.crop_size = CROP_SIZE
        self.use_numpy = use_numpy
        self.use_det = use_det
        self.columns_as_lists = G_columns_as_lists

        # with open(args.all_data_path, 'r') as file:
        #     self.json_file = json.load(file)
        self.json_file = json_file
        self.data_list = self.json_file[mode]
        self.data_len = len(self.data_list)

        # mean, std = self.calculate_mean_std()
        # print(f"Mean: {mean}, Std: {std}")
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
        #                          std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        # ])
            # transforms.Normalize(mean=[0.0], std=[1.0])
            transforms.Normalize(mean=[76.8/255],  # Normalize with mean and
                                 std=[55/255])  # standard deviation for pre-trained models on ImageNet
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            # transforms.Normalize(mean=[0.0], std=[1.0])
            transforms.Normalize(mean=[76.8/255],  # Normalize with mean and
                                 std=[55/255])  # standard deviation for pre-trained models on ImageNet
        ])

        # train_augmentation_steps = [
        #     # 阶段1：空间变换（保持解剖结构合理性）
        #     (tio.Resize(target_shape=(224, 224, -1)), 'Resize\n(128x128x128)'),
        #     (tio.Crop(cropping=((224 - self.crop_size)//2, (224 - self.crop_size)//2, (224 - self.crop_size)//2, (224 - self.crop_size)//2, 0, 0)), 'CenterCrop\n(H,W中心裁剪)'),
        #     # (tio.RandomAffine(scales=(
        #     #     0.9, 1.1,
        #     #     0.9, 1.1,
        #     #     1.0, 1.0
        #     # ),
        #     #     degrees=(0, 0, 10),  # (绕H, 绕W, 绕D) 旋转角度
        #     #     translation=(0, 0, 0),  # (H, W, D) 平移像素
        #     #     isotropic=False,  # 允许各向异性缩放
        #     #     default_pad_value='minimum', p=0.0), 'RandomAffine\n(仿射变换)'),
        #     # (tio.RandomElasticDeformation(num_control_points=7), 'ElasticDeform\n(弹性变形)'),
        #     # 阶段2：MRI特异性伪影增强
        #     (tio.RandomMotion(degrees=5, translation=2, p=0.0), 'RandomMotion\n(运动伪影)'),
        #     (tio.RandomGhosting(num_ghosts=1, p=0.0), 'RandomGhosting\n(重影伪影)'),
        #     # (tio.RandomBiasField(coefficients=0.1), 'RandomBiasField\n(偏置场伪影)'),
        #     # 阶段3：强度变换
        #     (tio.RandomNoise(std=0.05, p=0.0), 'RandomNoise\n(σ=0.05)'),
        #     (tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.0), 'RandomGamma\n(伽马校正)'),
        #     # 阶段4：标准化（最后执行）
        #     (tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(1, 99)), 'RescaleIntensity\n(强度归一化)'),
        # ]
        #
        # test_augmentation_steps = [
        #     # 阶段1：空间变换（保持解剖结构合理性）
        #     (tio.Resize(target_shape=(224, 224, -1)), 'Resize\n(128x128x128)'),
        #     (tio.Crop(cropping=(
        #         (224 - self.crop_size)//2, (224 - self.crop_size)//2, (224 - self.crop_size)//2, (224 - self.crop_size)//2, 0, 0)),
        #      'CenterCrop\n(H,W中心裁剪)'),
        #     (tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(1, 99)), 'RescaleIntensity\n(强度归一化)'),
        # ]

        train_transform_monai = mtf.Compose(
            [mtf.ResizeWithPadOrCropd(keys=["image", "seg"],spatial_size=(-1, 224, 224)),
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.1, spatial_axes=(1, 2)),
                mtf.RandSpatialCropd(
                    keys=["image", "seg"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),

                mtf.NormalizeIntensityd(keys=["image"],),

                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.14),
                mtf.RandGaussianNoised(keys="image", std=0.01, prob=0.2),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform_monai = mtf.Compose(
            [
                mtf.ResizeWithPadOrCropd(keys=["image", "seg"],spatial_size=(-1, 224, 224)),
                mtf.RandSpatialCropd(
                    keys=["image", "seg"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.NormalizeIntensityd(keys=["image"],),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        if mode == 'train':
            if self.use_numpy:
                self.transform = train_transform_monai
            else:
                self.transform = train_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        elif mode == 'validation':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
            # self.transform = val_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        elif mode == 'test':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        set_track_meta(False)

    def calculate_mean_std(self):
        """
        Calculate the mean and standard deviation of the dataset.
        """
        sum_pixels = 0
        sum_squared_pixels = 0
        total_pixels = 0

        for idx in range(len(self.data_list)):
            try:
                image_path, _ = self.preprocess_image_text_squence(self.data_list[idx])
                if image_path != [[], [], []]:
                    for sax_sq_img_pths in image_path:
                        for item_image in sax_sq_img_pths:
                            image = np.load(str(os.path.join(self.data_root,
                                                             item_image.replace('\\', '/').replace('.png', '.npy'))))
                            sum_pixels += image.sum()
                            sum_squared_pixels += (image ** 2).sum()
                            total_pixels += image.size
            except Exception as e:
                print(f"Error processing index {idx}: {e}")

        mean = sum_pixels / total_pixels
        std = np.sqrt((sum_squared_pixels / total_pixels) - (mean ** 2))
        return mean, std

    def preprocess_image_text_squence(self, data):

        try:
            symbol = random.choice(data['SAX']['json_item_squence_symbol'])
            symbol_indice = data['SAX']['json_item_squence_symbol'].index(symbol)
        except:
            output_data_list = [[], [], []]
            output_text_list = ''
            return output_data_list, output_text_list

        excel_id = data['Text'].get('excel_id')
        try:
            output_text_list = 'Cardiac function: ' + self.columns_as_lists['心脏电影step2E'][excel_id]
        except:
            output_text_list = 'Cardiac function: Not found'

        sax_sqs = data['SAX']['json_item_squence'][symbol_indice]['sorted_file_names']

        ########Larry
        if len(sax_sqs) > 9:
            sax_sqs_select = sax_sqs[len(sax_sqs) // 2 - 5:len(sax_sqs) // 2 + 4]
        else:
            sax_sqs_select = sax_sqs

        if len(sax_sqs_select) < 3:
            sax_pakge_len = 1
        else:
            sax_pakge_len = len(sax_sqs_select) // 3
        ########Larry

        sax_sq_names = []
        sax_sq_ids = []

        # sax_sq1 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [1,2,3]]))
        sax_sq1 = sax_sqs_select[:sax_pakge_len]
        if len(sax_sq1) > 0:
            sax_sq_names.append(random.choice(sax_sq1))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        # sax_sq2 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [4,5,6]]))
        sax_sq2 = sax_sqs_select[sax_pakge_len:2 * sax_pakge_len]
        if len(sax_sq2) > 0:
            sax_sq_names.append(random.choice(sax_sq2))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        # sax_sq3 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [7,8,9]]))
        sax_sq3 = sax_sqs_select[2 * sax_pakge_len:]
        if len(sax_sq3) > 0:
            sax_sq_names.append(random.choice(sax_sq3))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        selected_data_path = []
        for sax_sq_id in sax_sq_ids:
            if sax_sq_id is None:
                selected_data_path.append([])
            else:
                img_paths = data['SAX']['json_item_squence'][symbol_indice]['json_item_slice'][sax_sq_id]['Image_path']
                max_frame_num = min(30, len(img_paths))
                # start_frame = random.randint(0, len(img_paths) - max_frame_num)
                # img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # if self.mode == 'train':
                #     max_frame_num = min(30, len(img_paths))
                #     start_frame = random.randint(0, len(img_paths) - max_frame_num)
                #     img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # else:
                #     img_paths = img_paths  # [0:30] #[::frame_interval]
                img_paths = img_paths[:max_frame_num][::FRAME_INTERVAL]
                selected_data_path.append(img_paths)
        output_data_list = selected_data_path  # [ [sq1_img_paths,sq2_img_paths,sq3_img_paths]]
        return output_data_list, output_text_list

    def preprocess_image_text_squence_numpy(self, data):

        try:
            symbol = random.choice(data['SAX']['json_item_squence_symbol'])
            symbol_indice = data['SAX']['json_item_squence_symbol'].index(symbol)
        except:
            output_data_list = [[], [], []]
            output_text_list = ''
            return output_data_list, output_text_list
        excel_id = data['Text'].get('excel_id')
        try:
            output_text_list = 'Cine: '+ self.columns_as_lists['心脏形态step2E'][excel_id]
        except:
            output_text_list = 'Cine: No finding is observed.'
        # if data['Text'].get('excel_id') is None:
        #     output_text_list = 'Cine: No finding is observed.'
        # else:
        #     output_text_list = 'Cine: ' + data['Text']['text_Findings_SAX']

        sax_sqs = data['SAX']['json_item_squence'][symbol_indice]['sorted_file_names']
        # min_frame_num = min(25,min(data['SAX']['json_item_squence'][0]['number_squence_list']))
        # if min_frame_num < 25:
        #     import pdb;pdb.set_trace()

        ########Larry
        if len(sax_sqs) > 9:
            sax_sqs_select = sax_sqs[len(sax_sqs) // 2 - 5:len(sax_sqs) // 2 + 4]
        else:
            sax_sqs_select = sax_sqs

        if len(sax_sqs_select) < 3:
            sax_pakge_len = 1
        else:
            sax_pakge_len = len(sax_sqs_select) // 3
        ########Larry

        sax_sq_names = []
        sax_sq_ids = []

        # sax_sq1 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [1,2,3]]))
        sax_sq1 = sax_sqs_select[:sax_pakge_len]
        if len(sax_sq1) > 0:
            sax_sq_names.append(random.choice(sax_sq1))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        # sax_sq2 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [4,5,6]]))
        sax_sq2 = sax_sqs_select[sax_pakge_len:2 * sax_pakge_len]
        if len(sax_sq2) > 0:
            sax_sq_names.append(random.choice(sax_sq2))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        # sax_sq3 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [7,8,9]]))
        sax_sq3 = sax_sqs_select[2 * sax_pakge_len:]
        if len(sax_sq3) > 0:
            sax_sq_names.append(random.choice(sax_sq3))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        selected_data_path = []
        for sax_sq_id in sax_sq_ids:
            if sax_sq_id is None:
                selected_data_path.append([])
            else:
                img_paths = data['SAX']['json_item_squence'][symbol_indice]['json_item_slice'][sax_sq_id]['Image_path']
                max_frame_num = min(30, len(img_paths))
                start_frame = random.randint(0, len(img_paths) - max_frame_num)
                img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # if self.mode == 'train':
                #     max_frame_num = min(30, len(img_paths))
                #     start_frame = random.randint(0, len(img_paths) - max_frame_num)
                #     img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # else:
                #     img_paths = img_paths  # [0:30] #[::frame_interval]
                img_paths = img_paths[::FRAME_INTERVAL]
                selected_data_path.append(img_paths)
        output_data_list = selected_data_path  # [ [sq1_img_paths,sq2_img_paths,sq3_img_paths]]
        return output_data_list, output_text_list

    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer, special_token=f"<vsaxpatch>"):
        NON_VISION_TOKEN = -1
        cur_patch_indices = []
        img_tokens = []
        vision_patches = []
        patches = images
        n_deeps, n_rows, n_cols = patches.shape[1:]  # (1,14,14,3,5,16,16)
        if n_deeps % 3:
            pad_d = (3 - n_deeps % 3)
            n_deeps = (pad_d + n_deeps) // 3
        else:
            n_deeps =  n_deeps // 3
        n_rows = n_rows // 32
        n_cols = n_cols // 32
        n_patches = n_deeps * n_rows * n_cols
        # import pdb;pdb.set_trace()
        # patches = patches.view(n_patches, -1)

        # ---
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    img_tokens.append(special_token)
                    cur_patch_indices.append(
                        len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)

        # vision_patches = torch.Tensor(patches).bfloat16()
        # print(len(img_tokens),len(cur_patch_indices))
        assert len(img_tokens) == len(cur_patch_indices), f"{len(img_tokens)} != {len(cur_patch_indices)}"

        return img_tokens, cur_patch_indices#, vision_patches

    def __getitem__(self, idx):
        max_attempts = 100
        NON_VISION_TOKEN = -1
        for _ in range(max_attempts):
            try:
                image_path, text = self.preprocess_image_text_squence(self.data_list[idx])
                if image_path != [[], [], []]:
                    image_abs_path = image_path
                    vision_tokens = []
                    vision_patches = []
                    vision_patch_indices = []

                    special_token_list = [f"<vsaxpatch1>", f"<vsaxpatch2>", f"<vsaxpatch3>", ]
                    special_image_list = []
                    for sax_sq_id, sax_sq_img_pths in enumerate(image_path):  # 遍历不同的切面，每个切面自带时序
                        # print(sax_sq_id)
                        if len(sax_sq_img_pths) == 0:
                            continue
                        image_list = []
                        mask_list = []
                        if self.use_numpy:
                            for item_image in sax_sq_img_pths:
                                image = np.load(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.png','.npy'))))
                                # H, W
                                image_list.append(image[np.newaxis, ...])
                                if self.use_seg:
                                    image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root,
                                                                            item_image.replace('\\', '/').replace(
                                                                                '.png',
                                                                                '.npy'))))
                                    # image_seg_Rv = np.load(str(os.path.join(self.args.seg_Rv_root,
                                    #                                         item_image.replace('\\', '/').replace('.png',
                                    #                                                                               '.npy'))))
                                    image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root,
                                                                             item_image.replace('\\', '/').replace(
                                                                                 '.png',
                                                                                 '.npy'))))

                                    image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
                                    # image_seg_Rv = self.mask_transform(torch.from_numpy(image_seg_Rv))
                                    image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
                                    mask = torch.cat([image_seg_Lv, image_seg_MYO])
                                    mask_list.append(mask)




                            images = np.squeeze(np.array(image_list))
                            images = images[np.newaxis, ...]
                            mask_list = np.transpose(np.array(mask_list), (1, 0, 2, 3))
                            item = {
                                "image": images,
                                "seg": mask_list,
                            }

                            it = self.transform(item)

                            image = it['image']
                            seg = it['seg']  # C*D*H*W
                            images = torch.cat([image, seg], dim=0)#.permute(0, 3, 1, 2)
                            # images = torch.cat(image_list, dim=0).transpose(0, 1)
                        else:
                            for item_image in sax_sq_img_pths:

                                image = Image.open(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.npy','.png'))))
                                if self.use_det:
                                    box_list_npy = np.load(str(os.path.join(self.det_root,
                                                                            item_image.replace('\\', '/').replace(
                                                                                '.png', '.npy'))))
                                    LV_box = box_list_npy[0]  # x_center, y_center, box_w, box_h
                                    MYO_box = box_list_npy[1]  # x_center, y_center, box_w, box_h
                                    RV_box = box_list_npy[2]  # x_center, y_center, box_w, box_h
                                    # Calculate the union of the three boxes
                                    image_array = np.array(image)  # Convert to NumPy array
                                    h, w = image_array.shape[:2]
                                    x_min = min(LV_box[0] - LV_box[2] / 2, MYO_box[0] - MYO_box[2] / 2,
                                                RV_box[0] - RV_box[2] / 2) * w
                                    x_max = max(LV_box[0] + LV_box[2] / 2, MYO_box[0] + MYO_box[2] / 2,
                                                RV_box[0] + RV_box[2] / 2) * w
                                    y_min = min(LV_box[1] - LV_box[3] / 2, MYO_box[1] - MYO_box[3] / 2,
                                                RV_box[1] - RV_box[3] / 2) * h
                                    y_max = max(LV_box[1] + LV_box[3] / 2, MYO_box[1] + MYO_box[3] / 2,
                                                RV_box[1] + RV_box[3] / 2) * h

                                    # Create a mask for the union box
                                    mask = np.zeros_like(image_array, dtype=image_array.dtype)
                                    mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
                                    image = image_array * mask

                                image = self.transform(image) # C H W
                                if self.use_seg:
                                    image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    # image_seg_Rv = np.load(str(os.path.join(self.args.seg_Rv_root,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    # pil_mask = Image.fromarray(image_seg_Lv.astype(np.uint8), mode='L')
                                    # 1,H,W
                                    image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
                                    # image_seg_Rv = self.mask_transform(torch.from_numpy(image_seg_Rv))
                                    image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
                                    image = torch.cat([image, image_seg_Lv, image_seg_MYO])
                                image_list.append(image[np.newaxis, ...])

                            images = torch.cat(image_list, dim=0).transpose(0, 1)
                        # img_patches_vst = preprocess_image3D_vst(images)
                        # img_patches = preprocess_image3D(images)

                        cur_tokens, cur_patch_indices = self.prepare_inputs_img(
                            images, text, self.tokenizer, special_token=special_token_list[sax_sq_id])
                        # image_re = reconstruct_image3D(img_patches, images.size(), patch_size=PATCH_SIZE, patch_size_c=PATCH_SIZE_C)
                        special_image_list.append(images)
                        vision_tokens.extend(cur_tokens)
                        vision_tokens.append(f"<vslice_sep>")
                        # vision_patches.append(cur_vision_patches)
                        update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                            NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                                cur_patch_indices]
                        vision_patch_indices.extend(update_patch_indices)
                        vision_patch_indices.append(NON_VISION_TOKEN)
                        # print(sax_sq_id,len(vision_tokens),len(vision_patch_indices))

                    vision_tokens.append(f"<vsax_sep>")
                    vision_patch_indices.append(NON_VISION_TOKEN)
                    # import pdb;pdb.set_trace()
                    ret = {
                        "input_ids": vision_tokens,
                        "vision_patches": vision_patches,
                        "vision_patch_indices": vision_patch_indices,
                        "text": text,
                        "org_image_list":special_image_list
                        # 'org_'
                        # "text": text,
                        # "image_path": image_path,
                        # "question": question
                    }
                    return ret
                else:
                    ret = {
                        "input_ids": [],
                        "vision_patches": None,
                        "vision_patch_indices": [],
                        "text": "",
                        "org_image_list":[]
                    }
                    return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                # idx = random.randint(0, len(self.data_list) - 1)
                ret = {
                    "input_ids": [],
                    "vision_patches": None,
                    "vision_patch_indices": [],
                    "text": "",
                    "org_image_list":[]
                }
                return ret


class SAXCineDataset_CMR_3DFilm_vst_ALL(Dataset):
    def __init__(self, args, data_root, tokenizer, json_file,
                 G_columns_as_lists, mode="train", random_slize=False, use_seg=False, use_numpy = False, use_det = False):
        self.args = args
        # self.data_root = args.data_root
        self.data_root = data_root['image']
        self.seg_Lv_root = data_root['LV_mask']
        self.seg_MYO_root = data_root['MYO_mask']
        self.det_root  = data_root['DET_mask']
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.use_seg = use_seg
        self.max_position_embeddings = 4096
        self.crop_size = CROP_SIZE
        self.use_numpy = use_numpy
        self.use_det = use_det
        self.columns_as_lists = G_columns_as_lists

        # with open(args.all_data_path, 'r') as file:
        #     self.json_file = json.load(file)
        self.json_file = json_file
        self.data_list = self.json_file[mode]
        self.data_len = len(self.data_list)

        # mean, std = self.calculate_mean_std()
        # print(f"Mean: {mean}, Std: {std}")
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
        #                          std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        # ])
            # transforms.Normalize(mean=[0.0], std=[1.0])
            transforms.Normalize(mean=[76.8/255],  # Normalize with mean and
                                 std=[55/255])  # standard deviation for pre-trained models on ImageNet
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            # transforms.Normalize(mean=[0.0], std=[1.0])
            transforms.Normalize(mean=[76.8/255],  # Normalize with mean and
                                 std=[55/255])  # standard deviation for pre-trained models on ImageNet
        ])

        train_transform_monai = mtf.Compose(
            [mtf.ResizeWithPadOrCropd(keys=["image", "seg"],spatial_size=(-1, 224, 224)),
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.1, spatial_axes=(1, 2)),
                mtf.RandSpatialCropd(
                    keys=["image", "seg"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),

                mtf.NormalizeIntensityd(keys=["image"],),

                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.14),
                mtf.RandGaussianNoised(keys="image", std=0.01, prob=0.2),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform_monai = mtf.Compose(
            [
                mtf.ResizeWithPadOrCropd(keys=["image", "seg"],spatial_size=(-1, 224, 224)),
                mtf.RandSpatialCropd(
                    keys=["image", "seg"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.NormalizeIntensityd(keys=["image"],),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        if mode == 'train':
            if self.use_numpy:
                self.transform = train_transform_monai
            else:
                self.transform = train_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        elif mode == 'validation':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
            # self.transform = val_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        elif mode == 'test':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        set_track_meta(False)

    def calculate_mean_std(self):
        """
        Calculate the mean and standard deviation of the dataset.
        """
        sum_pixels = 0
        sum_squared_pixels = 0
        total_pixels = 0

        for idx in range(len(self.data_list)):
            try:
                image_path, _ = self.preprocess_image_text_squence(self.data_list[idx])
                if image_path != [[], [], []]:
                    for sax_sq_img_pths in image_path:
                        for item_image in sax_sq_img_pths:
                            image = np.load(str(os.path.join(self.data_root,
                                                             item_image.replace('\\', '/').replace('.png', '.npy'))))
                            sum_pixels += image.sum()
                            sum_squared_pixels += (image ** 2).sum()
                            total_pixels += image.size
            except Exception as e:
                print(f"Error processing index {idx}: {e}")

        mean = sum_pixels / total_pixels
        std = np.sqrt((sum_squared_pixels / total_pixels) - (mean ** 2))
        return mean, std

    def preprocess_image_text_squence(self, data):

        try:
            symbol = random.choice(data['SAX']['json_item_squence_symbol'])
            symbol_indice = data['SAX']['json_item_squence_symbol'].index(symbol)
        except:
            output_data_list = [[], [], []]
            output_text_list = ''
            return output_data_list, output_text_list

        excel_id = data['Text'].get('excel_id')
        try:
            output_text_list = 'Cardiac function: ' + self.columns_as_lists['心脏电影step2E'][excel_id]
        except:
            output_text_list = 'Cardiac function: Not found'

        sax_sqs = data['SAX']['json_item_squence'][symbol_indice]['sorted_file_names']

        ########Larry
        if len(sax_sqs) > 9:
            sax_sqs_select = sax_sqs[len(sax_sqs) // 2 - 5:len(sax_sqs) // 2 + 4]
        else:
            sax_sqs_select = sax_sqs


        sax_sq_names = []
        sax_sq_ids = []

        # sax_sq1 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [1,2,3]]))
        for sax_sq1 in sax_sqs_select:
            sax_sq_names.append(sax_sq1)
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))



        selected_data_path = []
        for sax_sq_id in sax_sq_ids:
            if sax_sq_id is None:
                selected_data_path.append([])
            else:
                img_paths = data['SAX']['json_item_squence'][symbol_indice]['json_item_slice'][sax_sq_id]['Image_path']
                max_frame_num = min(30, len(img_paths))
                # start_frame = random.randint(0, len(img_paths) - max_frame_num)
                img_paths = img_paths[:max_frame_num]  # [::frame_interval]
                # if self.mode == 'train':
                #     max_frame_num = min(30, len(img_paths))
                #     start_frame = random.randint(0, len(img_paths) - max_frame_num)
                #     img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # else:
                #     img_paths = img_paths  # [0:30] #[::frame_interval]
                img_paths = img_paths[::2]
                selected_data_path.append(img_paths)
        output_data_list = selected_data_path  # [ [sq1_img_paths,sq2_img_paths,sq3_img_paths]]
        return output_data_list, output_text_list

    def preprocess_image_text_squence_numpy(self, data):

        try:
            symbol = random.choice(data['SAX']['json_item_squence_symbol'])
            symbol_indice = data['SAX']['json_item_squence_symbol'].index(symbol)
        except:
            output_data_list = [[], [], []]
            output_text_list = ''
            return output_data_list, output_text_list
        excel_id = data['Text'].get('excel_id')
        try:
            output_text_list = 'Cine: '+ self.columns_as_lists['心脏形态step2E'][excel_id]
        except:
            output_text_list = 'Cine: No finding is observed.'
        # if data['Text'].get('excel_id') is None:
        #     output_text_list = 'Cine: No finding is observed.'
        # else:
        #     output_text_list = 'Cine: ' + data['Text']['text_Findings_SAX']

        sax_sqs = data['SAX']['json_item_squence'][symbol_indice]['sorted_file_names']
        # min_frame_num = min(25,min(data['SAX']['json_item_squence'][0]['number_squence_list']))
        # if min_frame_num < 25:
        #     import pdb;pdb.set_trace()

        ########Larry
        if len(sax_sqs) > 9:
            sax_sqs_select = sax_sqs[len(sax_sqs) // 2 - 5:len(sax_sqs) // 2 + 4]
        else:
            sax_sqs_select = sax_sqs

        if len(sax_sqs_select) < 3:
            sax_pakge_len = 1
        else:
            sax_pakge_len = len(sax_sqs_select) // 3
        ########Larry

        sax_sq_names = []
        sax_sq_ids = []

        # sax_sq1 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [1,2,3]]))
        sax_sq1 = sax_sqs_select[:sax_pakge_len]
        if len(sax_sq1) > 0:
            sax_sq_names.append(random.choice(sax_sq1))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        # sax_sq2 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [4,5,6]]))
        sax_sq2 = sax_sqs_select[sax_pakge_len:2 * sax_pakge_len]
        if len(sax_sq2) > 0:
            sax_sq_names.append(random.choice(sax_sq2))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        # sax_sq3 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [7,8,9]]))
        sax_sq3 = sax_sqs_select[2 * sax_pakge_len:]
        if len(sax_sq3) > 0:
            sax_sq_names.append(random.choice(sax_sq3))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        selected_data_path = []
        for sax_sq_id in sax_sq_ids:
            if sax_sq_id is None:
                selected_data_path.append([])
            else:
                img_paths = data['SAX']['json_item_squence'][symbol_indice]['json_item_slice'][sax_sq_id]['Image_path']
                max_frame_num = min(30, len(img_paths))
                start_frame = random.randint(0, len(img_paths) - max_frame_num)
                img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # if self.mode == 'train':
                #     max_frame_num = min(30, len(img_paths))
                #     start_frame = random.randint(0, len(img_paths) - max_frame_num)
                #     img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # else:
                #     img_paths = img_paths  # [0:30] #[::frame_interval]
                img_paths = img_paths[::FRAME_INTERVAL]
                selected_data_path.append(img_paths)
        output_data_list = selected_data_path  # [ [sq1_img_paths,sq2_img_paths,sq3_img_paths]]
        return output_data_list, output_text_list

    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer, special_token=f"<vsaxpatch>", special_token_T = "<vsaxpatch_T>"):
        NON_VISION_TOKEN = -1
        cur_patch_indices = []
        img_tokens = []
        vision_patches = []
        patches = images
        n_T= patches.shape[2]
        n_deeps= patches.shape[0]  # (1,14,14,3,5,16,16)
        n_rows = patches.shape[-2]  # (1,14,14,3,5,16,16)
        n_cols = patches.shape[-1]  # (1,14,14,3,5,16,16)
        if n_deeps % 3:
            pad_d = (3 - n_deeps % 3)
            n_deeps = (pad_d + n_deeps) // 3
        else:
            n_deeps =  n_deeps // 3
        n_rows = n_rows // 32
        n_cols = n_cols // 32
        n_patches = n_deeps * n_rows * n_cols
        # import pdb;pdb.set_trace()
        # patches = patches.view(n_patches, -1)

        # ---
        for T_idx in range(n_T):
            img_tokens.append(special_token_T)
            cur_patch_indices.append(T_idx)

        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    img_tokens.append(special_token)
                    cur_patch_indices.append(
                        n_T + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)

        # vision_patches = torch.Tensor(patches).bfloat16()
        # print(len(img_tokens),len(cur_patch_indices))
        assert len(img_tokens) == len(cur_patch_indices), f"{len(img_tokens)} != {len(cur_patch_indices)}"

        return img_tokens, cur_patch_indices#, vision_patches

    def __getitem__(self, idx):
        max_attempts = 100
        NON_VISION_TOKEN = -1
        for _ in range(max_attempts):
            try:
                image_path, text = self.preprocess_image_text_squence(self.data_list[idx])
                if image_path != [[], [], []]:
                    image_abs_path = image_path
                    vision_tokens = []
                    vision_patches = []
                    vision_patch_indices = []

                    # special_token_list = [f"<vsaxpatch1>", f"<vsaxpatch2>", f"<vsaxpatch3>", ]
                    special_image_list = []
                    for sax_sq_id, sax_sq_img_pths in enumerate(image_path):  # 遍历不同的切面，每个切面自带时序
                        # print(sax_sq_id)
                        if len(sax_sq_img_pths) == 0:
                            continue
                        image_list = []
                        mask_list = []
                        if self.use_numpy:
                            for item_image in sax_sq_img_pths:
                                image = np.load(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.png','.npy'))))
                                # H, W
                                image_list.append(image[np.newaxis, ...])
                                if self.use_seg:
                                    image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root,
                                                                            item_image.replace('\\', '/').replace(
                                                                                '.png',
                                                                                '.npy'))))
                                    # image_seg_Rv = np.load(str(os.path.join(self.args.seg_Rv_root,
                                    #                                         item_image.replace('\\', '/').replace('.png',
                                    #                                                                               '.npy'))))
                                    image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root,
                                                                             item_image.replace('\\', '/').replace(
                                                                                 '.png',
                                                                                 '.npy'))))

                                    image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
                                    # image_seg_Rv = self.mask_transform(torch.from_numpy(image_seg_Rv))
                                    image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
                                    mask = torch.cat([image_seg_Lv, image_seg_MYO])
                                    mask_list.append(mask)




                            images = np.squeeze(np.array(image_list))
                            images = images[np.newaxis, ...]
                            mask_list = np.transpose(np.array(mask_list), (1, 0, 2, 3))
                            item = {
                                "image": images,
                                "seg": mask_list,
                            }

                            it = self.transform(item)

                            image = it['image']
                            seg = it['seg']  # C*D*H*W
                            images = torch.cat([image, seg], dim=0)#.permute(0, 3, 1, 2)
                            # images = torch.cat(image_list, dim=0).transpose(0, 1)
                        else:
                            for item_image in sax_sq_img_pths:

                                image = Image.open(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.npy','.png'))))
                                if self.use_det:
                                    box_list_npy = np.load(str(os.path.join(self.det_root,
                                                                            item_image.replace('\\', '/').replace(
                                                                                '.png', '.npy'))))
                                    LV_box = box_list_npy[0]  # x_center, y_center, box_w, box_h
                                    MYO_box = box_list_npy[1]  # x_center, y_center, box_w, box_h
                                    RV_box = box_list_npy[2]  # x_center, y_center, box_w, box_h
                                    # Calculate the union of the three boxes
                                    image_array = np.array(image)  # Convert to NumPy array
                                    h, w = image_array.shape[:2]
                                    x_min = min(LV_box[0] - LV_box[2] / 2, MYO_box[0] - MYO_box[2] / 2,
                                                RV_box[0] - RV_box[2] / 2) * w
                                    x_max = max(LV_box[0] + LV_box[2] / 2, MYO_box[0] + MYO_box[2] / 2,
                                                RV_box[0] + RV_box[2] / 2) * w
                                    y_min = min(LV_box[1] - LV_box[3] / 2, MYO_box[1] - MYO_box[3] / 2,
                                                RV_box[1] - RV_box[3] / 2) * h
                                    y_max = max(LV_box[1] + LV_box[3] / 2, MYO_box[1] + MYO_box[3] / 2,
                                                RV_box[1] + RV_box[3] / 2) * h

                                    # Create a mask for the union box
                                    mask = np.zeros_like(image_array, dtype=image_array.dtype)
                                    mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
                                    image = image_array * mask

                                image = self.transform(image) # C H W
                                if self.use_seg:
                                    image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    # image_seg_Rv = np.load(str(os.path.join(self.args.seg_Rv_root,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    # pil_mask = Image.fromarray(image_seg_Lv.astype(np.uint8), mode='L')
                                    # 1,H,W
                                    image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
                                    # image_seg_Rv = self.mask_transform(torch.from_numpy(image_seg_Rv))
                                    image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
                                    image = torch.cat([image, image_seg_Lv, image_seg_MYO])
                                image_list.append(image[np.newaxis, ...])

                            images = torch.cat(image_list, dim=0).transpose(0, 1)
                        # img_patches_vst = preprocess_image3D_vst(images)
                        # img_patches = preprocess_image3D(images)
                        special_image_list.append(images)

                        # print(sax_sq_id,len(vision_tokens),len(vision_patch_indices))

                    t_list = [special_image_list_i.size(1) for special_image_list_i in special_image_list]
                    t_i_max = most_common_number(t_list)
                    # print(t_i_max)
                    new_list = []
                    for x in special_image_list:
                        if x.size(1) == t_i_max:
                            new_list.append(x)
                        else:
                            pad_size = t_i_max - x.size(1)
                            if pad_size>0:
                                padded_x = torch.nn.functional.pad(
                                    x,
                                    (0, 0,  # W dimension (no padding)
                                     0, 0,  # H dimension (no padding)
                                     0, pad_size,  # T dimension (pad after)
                                     0, 0),  # C dimension (no padding)
                                    mode='constant',
                                    value=0
                                )
                            else:
                                padded_x = x[:,:t_i_max, :, :]
                            new_list.append(padded_x)
                    special_image_list = torch.stack(new_list)

                    cur_tokens, cur_patch_indices = self.prepare_inputs_img(
                        special_image_list, text, self.tokenizer, special_token="<vsaxpatch1>", special_token_T = "<vsaxpatch2>")
                    # image_re = reconstruct_image3D(img_patches, images.size(), patch_size=PATCH_SIZE, patch_size_c=PATCH_SIZE_C)

                    vision_tokens.extend(cur_tokens)
                    vision_tokens.append(f"<vslice_sep>")
                    # vision_patches.append(cur_vision_patches)
                    update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                        NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                            cur_patch_indices]
                    vision_patch_indices.extend(update_patch_indices)
                    vision_patch_indices.append(NON_VISION_TOKEN)
                    vision_tokens.append(f"<vsax_sep>")
                    vision_patch_indices.append(NON_VISION_TOKEN)
                    # import pdb;pdb.set_trace()
                    ret = {
                        "input_ids": vision_tokens,
                        "vision_patches": vision_patches,
                        "vision_patch_indices": vision_patch_indices,
                        "text": text,
                        "org_image_list":special_image_list
                        # 'org_'
                        # "text": text,
                        # "image_path": image_path,
                        # "question": question
                    }
                    return ret
                else:
                    ret = {
                        "input_ids": [],
                        "vision_patches": None,
                        "vision_patch_indices": [],
                        "text": "",
                        "org_image_list":None
                    }
                    return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                # idx = random.randint(0, len(self.data_list) - 1)
                ret = {
                    "input_ids": [],
                    "vision_patches": None,
                    "vision_patch_indices": [],
                    "text": "",
                    "org_image_list":[]
                }
                return ret


class SAXCineDataset_CMR_3DFilm_vst_location(Dataset):
    def __init__(self, args, data_root, tokenizer, json_file, G_columns_as_lists, mode="train", random_slize=False, use_seg=False, use_numpy = False):
        self.args = args
        # self.data_root = args.data_root
        self.data_root = data_root['image']
        self.seg_Lv_root = data_root['LV_mask']
        self.seg_MYO_root = data_root['MYO_mask']
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.use_seg = use_seg
        self.max_position_embeddings = 4096
        self.crop_size = CROP_SIZE
        self.use_numpy = use_numpy
        self.columns_as_lists = G_columns_as_lists

        # with open(args.all_data_path, 'r') as file:
        #     self.json_file = json.load(file)
        self.json_file = json_file
        self.data_list = self.json_file[mode]
        self.data_len = len(self.data_list)

        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])

        # train_augmentation_steps = [
        #     # 阶段1：空间变换（保持解剖结构合理性）
        #     (tio.Resize(target_shape=(224, 224, -1)), 'Resize\n(128x128x128)'),
        #     (tio.Crop(cropping=((224 - self.crop_size)//2, (224 - self.crop_size)//2, (224 - self.crop_size)//2, (224 - self.crop_size)//2, 0, 0)), 'CenterCrop\n(H,W中心裁剪)'),
        #     # (tio.RandomAffine(scales=(
        #     #     0.9, 1.1,
        #     #     0.9, 1.1,
        #     #     1.0, 1.0
        #     # ),
        #     #     degrees=(0, 0, 10),  # (绕H, 绕W, 绕D) 旋转角度
        #     #     translation=(0, 0, 0),  # (H, W, D) 平移像素
        #     #     isotropic=False,  # 允许各向异性缩放
        #     #     default_pad_value='minimum', p=0.0), 'RandomAffine\n(仿射变换)'),
        #     # (tio.RandomElasticDeformation(num_control_points=7), 'ElasticDeform\n(弹性变形)'),
        #     # 阶段2：MRI特异性伪影增强
        #     (tio.RandomMotion(degrees=5, translation=2, p=0.0), 'RandomMotion\n(运动伪影)'),
        #     (tio.RandomGhosting(num_ghosts=1, p=0.0), 'RandomGhosting\n(重影伪影)'),
        #     # (tio.RandomBiasField(coefficients=0.1), 'RandomBiasField\n(偏置场伪影)'),
        #     # 阶段3：强度变换
        #     (tio.RandomNoise(std=0.05, p=0.0), 'RandomNoise\n(σ=0.05)'),
        #     (tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.0), 'RandomGamma\n(伽马校正)'),
        #     # 阶段4：标准化（最后执行）
        #     (tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(1, 99)), 'RescaleIntensity\n(强度归一化)'),
        # ]
        #
        # test_augmentation_steps = [
        #     # 阶段1：空间变换（保持解剖结构合理性）
        #     (tio.Resize(target_shape=(224, 224, -1)), 'Resize\n(128x128x128)'),
        #     (tio.Crop(cropping=(
        #         (224 - self.crop_size)//2, (224 - self.crop_size)//2, (224 - self.crop_size)//2, (224 - self.crop_size)//2, 0, 0)),
        #      'CenterCrop\n(H,W中心裁剪)'),
        #     (tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(1, 99)), 'RescaleIntensity\n(强度归一化)'),
        # ]

        train_transform_monai = mtf.Compose(
            [mtf.ResizeWithPadOrCropd(keys=["image", "seg"],spatial_size=(-1, 224, 224)),
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.1, spatial_axes=(1, 2)),
                mtf.RandSpatialCropd(
                    keys=["image", "seg"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),

                mtf.NormalizeIntensityd(keys=["image"],),

                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.14),
                mtf.RandGaussianNoised(keys="image", std=0.01, prob=0.2),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform_monai = mtf.Compose(
            [
                mtf.ResizeWithPadOrCropd(keys=["image", "seg"],spatial_size=(-1, 224, 224)),
                mtf.RandSpatialCropd(
                    keys=["image", "seg"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.NormalizeIntensityd(keys=["image"],),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        if mode == 'train':
            if self.use_numpy:
                self.transform = train_transform_monai
            else:
                self.transform = train_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        elif mode == 'validation':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
            # self.transform = val_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        elif mode == 'test':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
            self.mask_transform = transforms.Compose([
                # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(self.crop_size),
            ])
        set_track_meta(False)



    def preprocess_image_text_squence_numpy(self, data):

        try:
            symbol = random.choice(data['SAX']['json_item_squence_symbol'])
            symbol_indice = data['SAX']['json_item_squence_symbol'].index(symbol)
        except:
            output_data_list = [[], [], []]
            output_text_list = ''
            return output_data_list, output_text_list
        excel_id = data['Text'].get('excel_id')
        try:
            output_text_list = 'Cine: '+ self.columns_as_lists['心脏形态step2E'][excel_id]
        except:
            output_text_list = 'Cine: No finding is observed.'
        # if data['Text'].get('excel_id') is None:
        #     output_text_list = 'Cine: No finding is observed.'
        # else:
        #     output_text_list = 'Cine: ' + data['Text']['text_Findings_SAX']

        sax_sqs = data['SAX']['json_item_squence'][symbol_indice]['sorted_file_names']
        # min_frame_num = min(25,min(data['SAX']['json_item_squence'][0]['number_squence_list']))
        # if min_frame_num < 25:
        #     import pdb;pdb.set_trace()

        ########Larry
        if len(sax_sqs) > 9:
            sax_sqs_select = sax_sqs[len(sax_sqs) // 2 - 5:len(sax_sqs) // 2 + 4]
        else:
            sax_sqs_select = sax_sqs

        if len(sax_sqs_select) < 3:
            sax_pakge_len = 1
        else:
            sax_pakge_len = len(sax_sqs_select) // 3
        ########Larry

        sax_sq_names = []
        sax_sq_ids = []

        # sax_sq1 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [1,2,3]]))
        sax_sq1 = sax_sqs_select[:sax_pakge_len]
        if len(sax_sq1) > 0:
            sax_sq_names.append(random.choice(sax_sq1))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        # sax_sq2 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [4,5,6]]))
        sax_sq2 = sax_sqs_select[sax_pakge_len:2 * sax_pakge_len]
        if len(sax_sq2) > 0:
            sax_sq_names.append(random.choice(sax_sq2))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        # sax_sq3 = list(set(sax_sqs) & set(['SAX_'+symbol+'_'+str(i) for i in [7,8,9]]))
        sax_sq3 = sax_sqs_select[2 * sax_pakge_len:]
        if len(sax_sq3) > 0:
            sax_sq_names.append(random.choice(sax_sq3))
            sax_sq_ids.append(sax_sqs.index(sax_sq_names[-1]))
        else:
            sax_sq_names.append(None)
            sax_sq_ids.append(None)

        selected_data_path = []
        for sax_sq_id in sax_sq_ids:
            if sax_sq_id is None:
                selected_data_path.append([])
            else:
                img_paths = data['SAX']['json_item_squence'][symbol_indice]['json_item_slice'][sax_sq_id]['Image_path']
                max_frame_num = min(30, len(img_paths))
                start_frame = random.randint(0, len(img_paths) - max_frame_num)
                img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # if self.mode == 'train':
                #     max_frame_num = min(30, len(img_paths))
                #     start_frame = random.randint(0, len(img_paths) - max_frame_num)
                #     img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
                # else:
                #     img_paths = img_paths  # [0:30] #[::frame_interval]
                img_paths = img_paths[::FRAME_INTERVAL]
                selected_data_path.append(img_paths)
        output_data_list = selected_data_path  # [ [sq1_img_paths,sq2_img_paths,sq3_img_paths]]
        return output_data_list, output_text_list

    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer, special_token=f"<vsaxpatch>"):
        NON_VISION_TOKEN = -1
        cur_patch_indices = []
        img_tokens = []
        vision_patches = []
        patches = images
        n_deeps, n_rows, n_cols = patches.shape[1:]  # (1,14,14,3,5,16,16)
        if n_deeps % 3:
            pad_d = (3 - n_deeps % 3)
            n_deeps = (pad_d + n_deeps) // 3
        else:
            n_deeps =  n_deeps // 3
        n_rows = n_rows // 32
        n_cols = n_cols // 32
        n_patches = n_deeps * n_rows * n_cols
        # import pdb;pdb.set_trace()
        # patches = patches.view(n_patches, -1)

        # ---
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    img_tokens.append(special_token)
                    cur_patch_indices.append(
                        len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)

        # vision_patches = torch.Tensor(patches).bfloat16()
        # print(len(img_tokens),len(cur_patch_indices))
        assert len(img_tokens) == len(cur_patch_indices), f"{len(img_tokens)} != {len(cur_patch_indices)}"

        return img_tokens, cur_patch_indices#, vision_patches

    # def __getitem__(self, idx):
    #     max_attempts = 100
    #     NON_VISION_TOKEN = -1
    #     for _ in range(max_attempts):
    #         try:
    #             data = self.data_list[idx]
    #             img_paths = data["image_path"]
    #             max_frame_num = min(30, len(img_paths))
    #             start_frame = random.randint(0, len(img_paths) - max_frame_num)
    #             img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
    #
    #             img_paths = img_paths[::FRAME_INTERVAL]
    #
    #             vision_tokens = []
    #             vision_patches = []
    #             vision_patch_indices = []
    #
    #             special_token_list = [f"<vsaxpatch1>", f"<vsaxpatch2>", f"<vsaxpatch3>", ]
    #             special_image_list = []
    #             sax_sq_img_pths = img_paths
    #             if len(sax_sq_img_pths) == 0:
    #                 continue
    #             image_list = []
    #             mask_list = []
    #             if self.use_numpy:
    #                 for item_image in sax_sq_img_pths:
    #                     image = np.load(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.png','.npy'))))
    #                     # H, W
    #                     image_list.append(image[np.newaxis, ...])
    #                     if self.use_seg:
    #                         image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root,
    #                                                                 item_image.replace('\\', '/').replace(
    #                                                                     '.png',
    #                                                                     '.npy'))))
    #                         # image_seg_Rv = np.load(str(os.path.join(self.args.seg_Rv_root,
    #                         #                                         item_image.replace('\\', '/').replace('.png',
    #                         #                                                                               '.npy'))))
    #                         image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root,
    #                                                                  item_image.replace('\\', '/').replace(
    #                                                                      '.png',
    #                                                                      '.npy'))))
    #
    #                         image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
    #                         # image_seg_Rv = self.mask_transform(torch.from_numpy(image_seg_Rv))
    #                         image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
    #                         mask = torch.cat([image_seg_Lv, image_seg_MYO])
    #                         mask_list.append(mask)
    #
    #
    #
    #
    #                 images = np.squeeze(np.array(image_list))
    #                 images = images[np.newaxis, ...]
    #                 mask_list = np.transpose(np.array(mask_list), (1, 0, 2, 3))
    #                 item = {
    #                     "image": images,
    #                     "seg": mask_list,
    #                 }
    #
    #                 it = self.transform(item)
    #
    #                 image = it['image']
    #                 seg = it['seg']  # C*D*H*W
    #                 images = torch.cat([image, seg], dim=0)#.permute(0, 3, 1, 2)
    #                 # images = torch.cat(image_list, dim=0).transpose(0, 1)
    #             else:
    #                 for item_image in sax_sq_img_pths:
    #                     image = Image.open(str(item_image))
    #                     image = self.transform(image) # C H W
    #                     if self.use_seg:
    #                         image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root,item_image.replace('\\', '/').replace('.png','.npy'))))
    #                         # image_seg_Rv = np.load(str(os.path.join(self.args.seg_Rv_root,item_image.replace('\\', '/').replace('.png','.npy'))))
    #                         image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root,item_image.replace('\\', '/').replace('.png','.npy'))))
    #                         # pil_mask = Image.fromarray(image_seg_Lv.astype(np.uint8), mode='L')
    #                         # 1,H,W
    #                         image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
    #                         # image_seg_Rv = self.mask_transform(torch.from_numpy(image_seg_Rv))
    #                         image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
    #                         image = torch.cat([image, image_seg_Lv, image_seg_MYO])
    #                     image_list.append(image[np.newaxis, ...])
    #
    #                 images = torch.cat(image_list, dim=0).transpose(0, 1)
    #                 # img_patches_vst = preprocess_image3D_vst(images)
    #                 # img_patches = preprocess_image3D(images)
    #
    #                 cur_tokens, cur_patch_indices = self.prepare_inputs_img(
    #                     images, text, self.tokenizer, special_token=special_token_list[sax_sq_id])
    #                 # image_re = reconstruct_image3D(img_patches, images.size(), patch_size=PATCH_SIZE, patch_size_c=PATCH_SIZE_C)
    #                 special_image_list.append(images)
    #                 vision_tokens.extend(cur_tokens)
    #                 vision_tokens.append(f"<vslice_sep>")
    #                 # vision_patches.append(cur_vision_patches)
    #                 update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
    #                     NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
    #                                         cur_patch_indices]
    #                 vision_patch_indices.extend(update_patch_indices)
    #                 vision_patch_indices.append(NON_VISION_TOKEN)
    #                 # print(sax_sq_id,len(vision_tokens),len(vision_patch_indices))
    #
    #                 vision_tokens.append(f"<vsax_sep>")
    #                 vision_patch_indices.append(NON_VISION_TOKEN)
    #                 # import pdb;pdb.set_trace()
    #                 ret = {
    #                     "input_ids": vision_tokens,
    #                     "vision_patches": vision_patches,
    #                     "vision_patch_indices": vision_patch_indices,
    #                     "text": text,
    #                     "org_image_list":special_image_list
    #                     # 'org_'
    #                     # "text": text,
    #                     # "image_path": image_path,
    #                     # "question": question
    #                 }
    #                 return ret
    #             else:
    #                 ret = {
    #                     "input_ids": [],
    #                     "vision_patches": None,
    #                     "vision_patch_indices": [],
    #                     "text": "",
    #                     "org_image_list":[]
    #                 }
    #                 return ret
    #
    #         except Exception as e:
    #             print(f"Error in __getitem__ at index {idx}: {e}")
    #             # idx = random.randint(0, len(self.data_list) - 1)
    #             ret = {
    #                 "input_ids": [],
    #                 "vision_patches": None,
    #                 "vision_patch_indices": [],
    #                 "text": "",
    #                 "org_image_list":[]
    #             }
    #             return ret


class FCHCineDataset_CMR_2DFilm_vst(Dataset):
    def __init__(self, args, data_root, tokenizer, json_file, G_columns_as_lists, mode="train", random_slize=False, use_numpy = False):
        self.args = args
        self.data_root = data_root['image']
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.max_position_embeddings = 4096
        # with open(args.all_data_path, 'r') as file:
        #     self.json_file = json.load(file)
        self.json_file = json_file
        self.data_list = self.json_file[mode]
        self.data_len = len(self.data_list)
        self.use_numpy = use_numpy

        self.columns_as_lists = G_columns_as_lists
        self.crop_size = CROP_SIZE
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])

        train_transform_monai = mtf.Compose(
            [
                mtf.ResizeWithPadOrCropd(
                    keys=["image"],
                    spatial_size=(-1, IMAGE_SIZE, IMAGE_SIZE),  # 目标尺寸 (H, W)
                ),
                mtf.RandRotate90d(keys=["image"], prob=0.1, spatial_axes=(1, 2)),
                mtf.RandSpatialCropd(
                    keys=["image"],
                    roi_size=(-1,self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.RandFlipd(keys=["image"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image"], prob=0.10, spatial_axis=2),
                mtf.NormalizeIntensityd(
                    keys=["image"],
                ),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
                mtf.RandGaussianNoised(keys="image", std=0.01, prob=0.2),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                # mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform_monai = mtf.Compose(
            [
                mtf.ResizeWithPadOrCropd(
                    keys=["image"],
                    spatial_size=(-1, IMAGE_SIZE, IMAGE_SIZE),  # 目标尺寸 (H, W)
                ),
                mtf.RandSpatialCropd(
                    keys=["image"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.NormalizeIntensityd(
                    keys=["image"],
                ),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                # mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )
        if mode == 'train':
            if self.use_numpy:
                self.transform= train_transform_monai
            else:
                self.transform = train_transform
        elif mode == 'validation':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
        elif mode == 'test':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform

        set_track_meta(False)

    def preprocess_image_text_squence(self, data):
        frame_interval = FRAME_INTERVAL
        try:
            symbol = random.choice(data['4CH']['json_item_squence_symbol'])
            symbol_indice = data['4CH']['json_item_squence_symbol'].index(symbol)
        except:
            output_data_list = [[]]
            output_text_list = ''
            return output_data_list, output_text_list

        # if self.use_numpy:
        #     excel_id = data['Text'].get('excel_id')
        #     output_text_list = 'Cardiac morphology: ' + self.columns_as_lists['心脏形态step2E'][excel_id]
        # else:
        #     if data['Text'].get('text_Findings_4CH') is None or str(data['Text']['text_Findings_4CH']) == 'nan' or type(
        #             data['Text']['text_Findings_4CH']) in [int, float]:
        #         output_text_list = 'Cardiac Shape: No finding is observed.'
        #     else:
        #         output_text_list = 'Cardiac Shape: ' + data['Text']['text_Findings_4CH']
        try:
            excel_id = data['Text'].get('excel_id')
            output_text_list = 'Cardiac morphology: ' + self.columns_as_lists['心脏形态step2E'][excel_id]
        except:
            output_text_list = 'Cardiac morphology: Not found'
        # excel_id = data['Text'].get('excel_id')
        # output_text_list = 'Cardiac morphology: ' + self.columns_as_lists['心脏形态step2E'][excel_id]
        fch_sqs = data['4CH']['json_item_squence'][symbol_indice]['sorted_file_names']

        fch_sq_name = random.choice(fch_sqs)
        fch_sq_id = fch_sqs.index(fch_sq_name)

        selected_data_path = []

        img_paths = data['4CH']['json_item_squence'][symbol_indice]['json_item_slice'][fch_sq_id]['Image_path']
        # if self.mode == 'train':
        #     max_frame_num = min(30, len(img_paths))
        #     start_frame = random.randint(0, len(img_paths) - max_frame_num)
        #     img_paths = img_paths[start_frame:start_frame + max_frame_num]  # [::frame_interval]
        # else:
        #     img_paths = img_paths[0:30]  # [::frame_interval]
        max_frame_num = min(30, len(img_paths))
        img_paths = img_paths[:max_frame_num]
        img_paths = img_paths[::FRAME_INTERVAL]
        selected_data_path.append(img_paths)

        output_data_list = selected_data_path  # [ [sq1_img_paths,sq2_img_paths,sq3_img_paths]]
        return output_data_list, output_text_list

    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer, special_token=f"<vfchpatch>"):
        NON_VISION_TOKEN = -1
        cur_patch_indices = []
        img_tokens = []
        vision_patches = []
        patches = images
        n_deeps, n_rows, n_cols = patches.shape[1:]  # (1,14,14,3,5,16,16)
        if n_deeps % 3:
            pad_d = (3 - n_deeps % 3)
            n_deeps = (pad_d + n_deeps) // 3
        else:
            n_deeps = n_deeps // 3
        n_rows = n_rows // 32
        n_cols = n_cols // 32
        n_patches = n_deeps * n_rows * n_cols
        # patches = patches.view(n_patches, -1)

        # ---
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):

                    # if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                    #     img_tokens.append(f"<vframe_sep>")
                    #     cur_patch_indices.append(NON_VISION_TOKEN)
                    #
                    # if row_idx != 0 and col_idx == 0:  # when new row starts
                    #     img_tokens.append(f"<vrow_sep>")
                    #     cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(special_token)
                    cur_patch_indices.append(
                        len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)

        # vision_patches = torch.Tensor(patches).bfloat16()
        assert len(img_tokens) == len(cur_patch_indices), f"{len(img_tokens)} != {len(cur_patch_indices)}"

        return img_tokens, cur_patch_indices

    def __getitem__(self, idx):
        max_attempts = 100
        NON_VISION_TOKEN = -1
        # for _ in range(max_attempts):
        try:
            image_path, text = self.preprocess_image_text_squence(self.data_list[idx])
            if image_path != [[]]:
                image_abs_path = image_path
                vision_tokens = []
                vision_patches = []
                vision_patch_indices = []
                special_image = None
                for fch_sq_id, fch_sq_img_pths in enumerate(image_path):  # 遍历不同的切面，每个切面自带时序
                    if len(fch_sq_img_pths) == 0:
                        continue

                    image_list = []
                    if self.use_numpy:
                        for item_image in fch_sq_img_pths:
                            image = np.load(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.png','.npy'))))
                            # H, W
                            image_list.append(image[np.newaxis, ...])

                        images = np.squeeze(np.array(image_list))
                        images = images[np.newaxis, ...]
                        # mask_list = np.transpose(np.array(mask_list), (1, 0, 2, 3))
                        item = {
                            "image": images,
                        }
                        it = self.transform(item)
                        images = it['image']
                    else:
                        for item_image in fch_sq_img_pths:
                            image = Image.open(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.npy','.png'))))
                            image = self.transform(image)
                            image_list.append(image[np.newaxis, ...])
                        images = torch.cat(image_list, dim=0).transpose(0, 1)
                    # img_patches = preprocess_image3D(images)
                    special_image=images
                    cur_tokens, cur_patch_indices = self.prepare_inputs_img(
                        images, text, self.tokenizer)
                    vision_tokens.extend(cur_tokens)
                    # vision_patches.append(cur_vision_patches)
                    update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                        NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                            cur_patch_indices]
                    vision_patch_indices.extend(update_patch_indices)

                vision_tokens.append(f"<vfch_sep>")
                vision_patch_indices.append(NON_VISION_TOKEN)
                # import pdb;pdb.set_trace()
                ret = {
                    "input_ids": vision_tokens,
                    "vision_patches": vision_patches,
                    "vision_patch_indices": vision_patch_indices,
                    "text": text,
                    "org_image": special_image
                    # "text": text,
                    # "image_path": image_path,
                    # "question": question
                }
                return ret
            else:
                ret = {
                    "input_ids": [],
                    "vision_patches": None,
                    "vision_patch_indices": [],
                    "text": "",
                    "org_image":None

                }
                return ret

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            # idx = random.randint(0, len(self.data_list) - 1)
            ret = {
                "input_ids": [],
                "vision_patches": None,
                "vision_patch_indices": [],
                "text": "",
                "org_image":None
            }
            return ret


class LGEDataset_CMR_3D_ST(Dataset):
    def __init__(self, args, data_root, tokenizer, json_file, G_columns_as_lists,
                 mode="train", random_slize=False, use_seg=False, use_numpy = False, use_det = False):
        self.args = args
        self.data_root = data_root['image']
        self.seg_Lv_root_LGE = data_root['LV_mask']
        self.seg_MYO_root_LGE = data_root['MYO_mask']
        self.det_root = data_root['DET_mask']
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.use_seg = use_seg
        self.use_det = use_det
        self.crop_size = CROP_SIZE
        self.max_position_embeddings = 4096

        self.use_numpy = use_numpy
        self.columns_as_lists = G_columns_as_lists


        # with open(args.all_data_path, 'r') as file:
        #     self.json_file = json.load(file)
        self.json_file = json_file
        self.data_list = self.json_file[mode]
        self.data_len = len(self.data_list)

        train_transform_monai = mtf.Compose(
            [mtf.ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=(-1, 224, 224)),
             mtf.RandRotate90d(keys=["image", "seg"], prob=0.1, spatial_axes=(1, 2)),
             mtf.RandSpatialCropd(
                 keys=["image", "seg"],
                 roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                 random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                 random_size=False  # 固定裁剪尺寸
             ),
             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),

             mtf.NormalizeIntensityd(keys=["image"], ),

             mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
             mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.14),
             mtf.RandGaussianNoised(keys="image", std=0.01, prob=0.2),
             mtf.ToTensord(keys=["image"], dtype=torch.float),
             mtf.ToTensord(keys=["seg"], dtype=torch.int),
             ]
        )

        val_transform_monai = mtf.Compose(
            [
                mtf.ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=(-1, 224, 224)),
                mtf.RandSpatialCropd(
                    keys=["image", "seg"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.NormalizeIntensityd(keys=["image"], ),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            # transforms.Normalize(mean=[0.0], std=[1.0])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            # transforms.Normalize(mean=[0.0], std=[1.0])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])
        self.mask_transform = transforms.Compose([
            # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
        ])

        if mode == 'train':
            if self.use_numpy:
                self.transform = train_transform_monai
            else:
                self.transform = train_transform
            # self.transform = train_transform

        elif mode == 'validation':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
        elif mode == 'test':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform

        set_track_meta(False)

    def preprocess_image_text_squence(self, data):
        try:
            symbol = random.choice(data['LGE']['json_item_squence_symbol'])
            symbol_indice = data['LGE']['json_item_squence_symbol'].index(symbol)
        except:
            output_data_list = [[]]
            output_text_list = ''
            return output_data_list, output_text_list

        if self.use_numpy:
            excel_id = data['Text'].get('excel_id')
            output_text_list = 'LGE: ' + self.columns_as_lists['LGEstep2E'][excel_id]
        else:
            if data['Text'].get('text_Findings_LGE') is None or str(data['Text']['text_Findings_LGE']) == 'nan' or type(
                    data['Text']['text_Findings_LGE']) in [int, float]:
                output_text_list = 'LGE: No finding is observed.'
            else:
                output_text_list = 'LGE: ' + data['Text']['text_Findings_LGE']

        lge_sqs = data['LGE']['json_item_squence'][symbol_indice]['sorted_file_names']
        lge_num = min(6, len(lge_sqs))

        lge_indexed_list = list(enumerate(lge_sqs))

        # 随机抽取 10 个（索引，值）对
        sampled_pairs = random.sample(lge_indexed_list, lge_num)

        # 分离索引和值
        lge_sq_ids = [pair[0] for pair in sampled_pairs]
        lge_sq_name = [pair[1] for pair in sampled_pairs]
        selected_data_path = []

        img_paths = [data['LGE']['json_item_squence'][symbol_indice]['json_item_slice'][lge_sq_id]['Image_path'][0] for
                     lge_sq_id in lge_sq_ids]

        selected_data_path.append(img_paths)

        output_data_list = selected_data_path  # [ [sq1_img_paths,sq2_img_paths,sq3_img_paths]]
        return output_data_list, output_text_list

    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer, special_token=f"<vlgepatch>"):
        NON_VISION_TOKEN = -1
        cur_patch_indices = []
        img_tokens = []
        vision_patches = []
        patches = images
        n_deeps, n_rows, n_cols = patches.shape[1:]  # (1,14,14,3,5,16,16)
        # if n_deeps % 3:
        #     pad_d = (3 - n_deeps % 3)
        #     n_deeps = (pad_d + n_deeps) // 3
        # else:
        #     n_deeps = n_deeps // 3
        n_rows = n_rows // 32
        n_cols = n_cols // 32
        n_patches = n_deeps * n_rows * n_cols
        # patches = patches.view(n_patches, -1)

        # ---
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    # if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                    #     img_tokens.append(f"<vframe_sep>")
                    #     cur_patch_indices.append(NON_VISION_TOKEN)
                    #
                    # if row_idx != 0 and col_idx == 0:  # when new row starts
                    #     img_tokens.append(f"<vrow_sep>")
                    #     cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(special_token)
                    cur_patch_indices.append(
                        len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)

        # vision_patches = torch.Tensor(patches).bfloat16()

        assert len(img_tokens) == len(cur_patch_indices), f"{len(img_tokens)} != {len(cur_patch_indices)}"

        return img_tokens, cur_patch_indices

    def __getitem__(self, idx):
        max_attempts = 100
        NON_VISION_TOKEN = -1
        for _ in range(max_attempts):
            try:
                image_path, text = self.preprocess_image_text_squence(self.data_list[idx])
                if image_path != [[]]:
                    image_abs_path = image_path
                    vision_tokens = []
                    vision_patches = []
                    vision_patch_indices = []
                    special_image = None
                    for lge_sq_id, lge_sq_img_pths in enumerate(image_path):  # 遍历不同的切面，每个切面自带时序
                        if len(lge_sq_img_pths) == 0:
                            continue
                        image_list = []
                        mask_list = []
                        if self.use_numpy:
                            for item_image in lge_sq_img_pths:
                                image = np.load(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.png','.npy'))))
                                # H, W

                                if self.use_seg:
                                    image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root,
                                                                            item_image.replace('\\', '/').replace(
                                                                                '.png',
                                                                                '.npy'))))
                                    image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root,
                                                                             item_image.replace('\\', '/').replace(
                                                                                 '.png',
                                                                                 '.npy'))))

                                    image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
                                    image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
                                    mask = torch.cat([image_seg_Lv, image_seg_MYO])
                                    mask_list.append(mask)


                                image_list.append(image[np.newaxis, ...])
                            images = np.squeeze(np.array(image_list))
                            images = images[np.newaxis, ...]
                            mask_list = np.transpose(np.array(mask_list), (1, 0, 2, 3))
                            item = {
                                "image": images,
                                "seg": mask_list,
                            }
                            it = self.transform(item)
                            image = it['image']
                            seg = it['seg']  # C*D*H*W
                            images = torch.cat([image, seg], dim=0)#.permute(0, 3, 1, 2)
                        else:
                            for item_image in lge_sq_img_pths:
                                image = Image.open(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.npy','.png'))))
                                if self.use_det:
                                    box_list_npy = np.load(str(os.path.join(self.det_root,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    LV_box = box_list_npy[0] #x_center, y_center, box_w, box_h
                                    MYO_box = box_list_npy[1] #x_center, y_center, box_w, box_h
                                    RV_box = box_list_npy[2] #x_center, y_center, box_w, box_h
                                    # Calculate the union of the three boxes
                                    image_array = np.array(image)  # Convert to NumPy array
                                    h, w = image_array.shape[:2]
                                    x_min = min(LV_box[0] - LV_box[2] / 2, MYO_box[0] - MYO_box[2] / 2,
                                                RV_box[0] - RV_box[2] / 2)*w
                                    x_max = max(LV_box[0] + LV_box[2] / 2, MYO_box[0] + MYO_box[2] / 2,
                                                RV_box[0] + RV_box[2] / 2)*w
                                    y_min = min(LV_box[1] - LV_box[3] / 2, MYO_box[1] - MYO_box[3] / 2,
                                                RV_box[1] - RV_box[3] / 2)*h
                                    y_max = max(LV_box[1] + LV_box[3] / 2, MYO_box[1] + MYO_box[3] / 2,
                                                RV_box[1] + RV_box[3] / 2)*h

                                    # Create a mask for the union box
                                    mask = np.zeros_like(image_array, dtype=image_array.dtype)
                                    mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
                                    image = image_array * mask

                                image = self.transform(image) # C H W
                                if self.use_seg:
                                    image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root_LGE,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root_LGE,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
                                    image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
                                    image = torch.cat([image, image_seg_Lv, image_seg_MYO])


                                image_list.append(image[np.newaxis, ...])
                            images = torch.cat(image_list, dim=0).transpose(0, 1)

                        cur_tokens, cur_patch_indices = self.prepare_inputs_img(
                            images, text, self.tokenizer)
                        vision_tokens.extend(cur_tokens)
                        # vision_patches.append(cur_vision_patches)
                        update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                            NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                                cur_patch_indices]
                        vision_patch_indices.extend(update_patch_indices)
                        special_image = images

                    vision_tokens.append(f"<vlge_sep>")
                    vision_patch_indices.append(NON_VISION_TOKEN)

                    ret = {
                        "input_ids": vision_tokens,
                        "org_image": special_image,
                        "vision_patch_indices": vision_patch_indices,
                        "text": text
                        # "text": text,
                        # "image_path": image_path,
                        # "question": question
                    }
                    return ret
                else:
                    ret = {
                        "input_ids": [],
                        "org_image": None,
                        "vision_patch_indices": [],
                        "text": ""
                    }
                    return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                # idx = random.randint(0, len(self.data_list) - 1)
                ret = {
                    "input_ids": [],
                    "org_image": None,
                    "vision_patch_indices": [],
                    "text": ""
                }
                return ret

class LGEDataset_CMR_3D_vst(Dataset):
    def __init__(self, args, data_root, tokenizer, json_file, G_columns_as_lists,
                 mode="train", random_slize=False, use_seg=False, use_numpy = False, use_det = False):
        self.args = args
        self.data_root = data_root['image']
        self.seg_Lv_root_LGE = data_root['LV_mask']
        self.seg_MYO_root_LGE = data_root['MYO_mask']
        self.det_root = data_root['DET_mask']
        self.tokenizer = tokenizer
        self.mode = mode
        self.random_slize = random_slize
        self.use_seg = use_seg
        self.use_det = use_det
        self.crop_size = CROP_SIZE
        self.max_position_embeddings = 4096

        self.use_numpy = use_numpy
        self.columns_as_lists = G_columns_as_lists


        # with open(args.all_data_path, 'r') as file:
        #     self.json_file = json.load(file)
        self.json_file = json_file
        self.data_list = self.json_file[mode]
        self.data_len = len(self.data_list)

        train_transform_monai = mtf.Compose(
            [mtf.ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=(-1, 224, 224)),
             mtf.RandRotate90d(keys=["image", "seg"], prob=0.1, spatial_axes=(1, 2)),
             mtf.RandSpatialCropd(
                 keys=["image", "seg"],
                 roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                 random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                 random_size=False  # 固定裁剪尺寸
             ),
             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),

             mtf.NormalizeIntensityd(keys=["image"], ),

             mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
             mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.14),
             mtf.RandGaussianNoised(keys="image", std=0.01, prob=0.2),
             mtf.ToTensord(keys=["image"], dtype=torch.float),
             mtf.ToTensord(keys=["seg"], dtype=torch.int),
             ]
        )

        val_transform_monai = mtf.Compose(
            [
                mtf.ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=(-1, 224, 224)),
                mtf.RandSpatialCropd(
                    keys=["image", "seg"],
                    roi_size=(-1, self.crop_size, self.crop_size),  # 目标裁剪尺寸 (H, W, D)
                    random_center=False,  # 随机中心位置（若False则为严格中心裁剪）
                    random_size=False  # 固定裁剪尺寸
                ),
                mtf.NormalizeIntensityd(keys=["image"], ),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            # transforms.Normalize(mean=[0.0], std=[1.0])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
            # transforms.Normalize(mean=[0.0], std=[1.0])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                 std=[0.229, 0.224, 0.225])  # standard deviation for pre-trained models on ImageNet
        ])
        self.mask_transform = transforms.Compose([
            # transforms.ToTensor(),  # 对于numpy数组，ToTensor会将其转换为[C, H, W]，这里C=1
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(self.crop_size),
        ])

        if mode == 'train':
            if self.use_numpy:
                self.transform = train_transform_monai
            else:
                self.transform = train_transform
            # self.transform = train_transform

        elif mode == 'validation':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform
        elif mode == 'test':
            if self.use_numpy:
                self.transform = val_transform_monai
            else:
                self.transform = val_transform

        set_track_meta(False)

    def preprocess_image_text_squence(self, data):
        try:
            symbol = random.choice(data['LGE']['json_item_squence_symbol'])
            symbol_indice = data['LGE']['json_item_squence_symbol'].index(symbol)
        except:
            output_data_list = [[]]
            output_text_list = ''
            return output_data_list, output_text_list

        # if self.use_numpy:
        #     excel_id = data['Text'].get('excel_id')
        #     output_text_list = 'LGE: ' + self.columns_as_lists['LGEstep2E'][excel_id]
        # else:
        #     if data['Text'].get('text_Findings_LGE') is None or str(data['Text']['text_Findings_LGE']) == 'nan' or type(
        #             data['Text']['text_Findings_LGE']) in [int, float]:
        #         output_text_list = 'LGE: No finding is observed.'
        #     else:
        #         output_text_list = 'LGE: ' + data['Text']['text_Findings_LGE']
        try:
            excel_id = data['Text'].get('excel_id')
            output_text_list = 'LGE: ' + self.columns_as_lists['LGEstep2E'][excel_id]
        except:
            output_text_list = 'LGE: No finding is observed.'
        lge_sqs = data['LGE']['json_item_squence'][symbol_indice]['sorted_file_names']
        lge_num = min(10, len(lge_sqs))

        lge_indexed_list = list(enumerate(lge_sqs))

        # 随机抽取 10 个（索引，值）对
        sampled_pairs = lge_indexed_list[:lge_num]

        # 分离索引和值
        lge_sq_ids = [pair[0] for pair in sampled_pairs]
        lge_sq_name = [pair[1] for pair in sampled_pairs]
        selected_data_path = []

        img_paths = [data['LGE']['json_item_squence'][symbol_indice]['json_item_slice'][lge_sq_id]['Image_path'][0] for
                     lge_sq_id in lge_sq_ids]

        selected_data_path.append(img_paths)

        output_data_list = selected_data_path  # [ [sq1_img_paths,sq2_img_paths,sq3_img_paths]]
        return output_data_list, output_text_list

    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, inputs, tokenizer, special_token=f"<vlgepatch>"):
        NON_VISION_TOKEN = -1
        cur_patch_indices = []
        img_tokens = []
        vision_patches = []
        patches = images
        n_deeps, n_rows, n_cols = patches.shape[1:]  # (1,14,14,3,5,16,16)
        # if n_deeps % 3:
        #     pad_d = (3 - n_deeps % 3)
        #     n_deeps = (pad_d + n_deeps) // 3
        # else:
        #     n_deeps = n_deeps // 3
        n_rows = n_rows // 32
        n_cols = n_cols // 32
        n_patches = n_deeps * n_rows * n_cols
        # patches = patches.view(n_patches, -1)

        # ---
        for depth_idx in range(n_deeps):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    # if depth_idx != 0 and row_idx == 0 and col_idx == 0:  # when new row starts
                    #     img_tokens.append(f"<vframe_sep>")
                    #     cur_patch_indices.append(NON_VISION_TOKEN)
                    #
                    # if row_idx != 0 and col_idx == 0:  # when new row starts
                    #     img_tokens.append(f"<vrow_sep>")
                    #     cur_patch_indices.append(NON_VISION_TOKEN)

                    img_tokens.append(special_token)
                    cur_patch_indices.append(
                        len(vision_patches) + depth_idx * n_rows * n_cols + row_idx * n_cols + col_idx)

        # vision_patches = torch.Tensor(patches).bfloat16()

        assert len(img_tokens) == len(cur_patch_indices), f"{len(img_tokens)} != {len(cur_patch_indices)}"

        return img_tokens, cur_patch_indices

    def __getitem__(self, idx):
        max_attempts = 100
        NON_VISION_TOKEN = -1
        for _ in range(max_attempts):
            try:
                image_path, text = self.preprocess_image_text_squence(self.data_list[idx])
                if image_path != [[]]:

                    image_abs_path = image_path
                    vision_tokens = []
                    vision_patches = []
                    vision_patch_indices = []
                    special_image = None
                    for lge_sq_id, lge_sq_img_pths in enumerate(image_path):  # 遍历不同的切面，每个切面自带时序
                        if len(lge_sq_img_pths) == 0:
                            continue
                        image_list = []
                        mask_list = []
                        if self.use_numpy:
                            for item_image in lge_sq_img_pths:
                                image = np.load(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.png','.npy'))))
                                # H, W

                                if self.use_seg:
                                    image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root,
                                                                            item_image.replace('\\', '/').replace(
                                                                                '.png',
                                                                                '.npy'))))
                                    image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root,
                                                                             item_image.replace('\\', '/').replace(
                                                                                 '.png',
                                                                                 '.npy'))))

                                    image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
                                    image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
                                    mask = torch.cat([image_seg_Lv, image_seg_MYO])
                                    mask_list.append(mask)


                                image_list.append(image[np.newaxis, ...])
                            images = np.squeeze(np.array(image_list))
                            images = images[np.newaxis, ...]
                            mask_list = np.transpose(np.array(mask_list), (1, 0, 2, 3))
                            item = {
                                "image": images,
                                "seg": mask_list,
                            }
                            it = self.transform(item)
                            image = it['image']
                            seg = it['seg']  # C*D*H*W
                            images = torch.cat([image, seg], dim=0)#.permute(0, 3, 1, 2)
                        else:
                            for item_image in lge_sq_img_pths:
                                image = Image.open(str(os.path.join(self.data_root, item_image.replace('\\', '/').replace('.npy','.png'))))
                                if self.use_det:
                                    box_list_npy = np.load(str(os.path.join(self.det_root,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    LV_box = box_list_npy[0] #x_center, y_center, box_w, box_h
                                    MYO_box = box_list_npy[1] #x_center, y_center, box_w, box_h
                                    RV_box = box_list_npy[2] #x_center, y_center, box_w, box_h
                                    # Calculate the union of the three boxes
                                    image_array = np.array(image)  # Convert to NumPy array
                                    h, w = image_array.shape[:2]
                                    x_min = min(LV_box[0] - LV_box[2] / 2, MYO_box[0] - MYO_box[2] / 2,
                                                RV_box[0] - RV_box[2] / 2)*w
                                    x_max = max(LV_box[0] + LV_box[2] / 2, MYO_box[0] + MYO_box[2] / 2,
                                                RV_box[0] + RV_box[2] / 2)*w
                                    y_min = min(LV_box[1] - LV_box[3] / 2, MYO_box[1] - MYO_box[3] / 2,
                                                RV_box[1] - RV_box[3] / 2)*h
                                    y_max = max(LV_box[1] + LV_box[3] / 2, MYO_box[1] + MYO_box[3] / 2,
                                                RV_box[1] + RV_box[3] / 2)*h

                                    # Create a mask for the union box
                                    mask = np.zeros_like(image_array, dtype=image_array.dtype)
                                    mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
                                    image = image_array * mask

                                image = self.transform(image) # C H W
                                if self.use_seg:
                                    image_seg_Lv = np.load(str(os.path.join(self.seg_Lv_root_LGE,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    image_seg_MYO = np.load(str(os.path.join(self.seg_MYO_root_LGE,item_image.replace('\\', '/').replace('.png','.npy'))))
                                    image_seg_Lv = self.mask_transform(torch.from_numpy(image_seg_Lv))
                                    image_seg_MYO = self.mask_transform(torch.from_numpy(image_seg_MYO))
                                    image = torch.cat([image, image_seg_Lv, image_seg_MYO])


                                image_list.append(image[np.newaxis, ...])
                            images = torch.cat(image_list, dim=0).transpose(0, 1)

                        cur_tokens, cur_patch_indices = self.prepare_inputs_img(
                            images, text, self.tokenizer)
                        vision_tokens.extend(cur_tokens)
                        # vision_patches.append(cur_vision_patches)
                        update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                            NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                                cur_patch_indices]
                        vision_patch_indices.extend(update_patch_indices)
                        special_image = images

                    vision_tokens.append(f"<vlge_sep>")
                    vision_patch_indices.append(NON_VISION_TOKEN)

                    ret = {
                        "input_ids": vision_tokens,
                        "org_image": special_image,
                        "vision_patch_indices": vision_patch_indices,
                        "text": text
                        # "text": text,
                        # "image_path": image_path,
                        # "question": question
                    }
                    return ret
                else:
                    ret = {
                        "input_ids": [],
                        "org_image": None,
                        "vision_patch_indices": [],
                        "text": ""
                    }
                    return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                # idx = random.randint(0, len(self.data_list) - 1)
                ret = {
                    "input_ids": [],
                    "org_image": None,
                    "vision_patch_indices": [],
                    "text": ""
                }
                return ret


class Detection_Dataset2(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train', multiple=1):
        self.args = args
        self.tokenizer = tokenizer
        self.description = description
        self.mode = mode
        self.max_position_embeddings = 4096
        # root_path = args.seg_data_path
        self.Image_size = args.image_s
        self.Patch_size = args.patch_s

        with open(args.seg_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]
        if self.mode == 'train':
            self.data_list = self.data_list * multiple
            random.shuffle(self.data_list)
        train_transform = mtf.Compose(
            [
                mtf.Resized(keys=["image", "seg"], spatial_size=[self.Image_size, self.Image_size, -1],
                            mode=['trilinear', 'nearest']),  # trilinear
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.25, spatial_axes=(0, 1)),
                # mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                # mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                # mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.05, prob=0.1),
                mtf.RandShiftIntensityd(keys="image", offsets=0.05, prob=0.1),
                # mtf.RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.02),
                mtf.RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(1.5, 2.5)),
                mtf.RandHistogramShiftd(keys=["image"], prob=0.1, num_control_points=3),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.Resized(keys=["image", "seg"], spatial_size=[self.Image_size, self.Image_size, -1],
                            mode=['trilinear', 'nearest']),  # trilinear
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        # with open(args.term_dict_path, 'r') as file:
        #     self.term_dict = json.load(file)

        self.question_list = [
            "Can you detect the {} in this image?",
            "Can you detect {} in this image? Please output the mask.",
            "Please detect the {} in this image.",
            "What is {} in this image? Please respond with detection mask.",
            "What is {} in this image? Please output detection mask.",
            "Could you provide a detection for the {}?",
            "I need the {} detected from this image.",
            "Detection {} from this image and provide the mask, please.",
            "Please provide a detection mask for the {} in this image.",
            "Can you identify and detect the {} in this image?",
        ]

        self.answer_list = [
            "Sure, it is <DET>.",
        ]

        self.description_list = [
            "Description: {} Please answer and segment based on the above description.",
            "Definition: {} Please answer and segment based on the above definition.",
            "Description: {} Can you answer and segment it based on the above description or definition.",
            "Definition: {} Please output segmentation mask and answer based on the above description or definition.",
            "Provided description: {} Please segment accordingly.",
            "Given definition: {} Please provide segmentation and answer according to it.",
            "The description provided is: {} Now, segment it and provide your answer.",
            "Based on the provided definition: {} Please segment and provide your response.",
            "Describing the object as: {} Can you segment it accordingly?",
            "Defining it as: {} Now, segment and provide your answer.",
        ]

        self.answer_cls_list = [
            "It is {}, <DET>.",
            ]

        self.answer_no_cls_list = [
            "Sorry, there is no {}",
        ]

    def __len__(self):
        return len(self.data_list)

    def prepare_inputs_img(self, images, tokenizer, cls_id, target_masks, target_boxes):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        patches = images
        n_rows, n_cols = patches.shape[:2]
        n_patches = n_rows * n_cols
        patches = patches.view(n_patches, -1)

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx != 0 and col_idx == 0:  # when new row starts
                    img_tokens.append(f"<vrow_sep>")
                    cur_patch_indices.append(NON_VISION_TOKEN)
                img_tokens.append(f"<vpatch>")
                cur_patch_indices.append(len(vision_patches) + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)

        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)
        # print(f"cur_tokens: {cur_tokens}")
        # print(f"cur_attention_mask: {cur_attention_mask}")
        # print(f"cur_patch_indices: {cur_patch_indices}")
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))

        cls_list = ['left ventricle (LV)',
                    'left ventricular myocardium (MYO)',
                    'right ventricle (RV)']
        # cls = 'left ventricle (LV), the left ventricular myocardium (MYO), and the right ventricle (RV)'

        question_temple = random.choice(self.question_list)
        question = question_temple.format(cls_list[cls_id])

        # 你的已生成的 target_masks, target_boxes
        # cls_id: 当前类别索引，cls_list：类别名称
        has_obj = target_masks[cls_id] > 0  # 当前类别是否存在

        question_temple = random.choice(self.question_list)
        question = question_temple.format(cls_list[cls_id])

        if has_obj:
            answer = random.choice(self.answer_cls_list).format(cls_list[cls_id])
            cur_target_boxes = target_boxes[cls_id]  # 仅返回当前类别的 box
            cur_target_mask = target_masks[cls_id]  # 仅当前类别
            answer = answer + "<DET>" + end_token
        else:
            answer = random.choice(self.answer_no_cls_list).format(cls_list[cls_id])
            cur_target_boxes =  torch.zeros(4)
            cur_target_mask = torch.zeros(1)
            answer = answer +  end_token

        # 下游你可以使用 cur_target_boxes, cur_target_mask 做后续计算
        # question, answer 会根据实际是否存在做出不同回答


        # answer = random.choice(self.answer_cls_list).format(cls_list[cls_id])

        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))


        _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend(cur_tokens)
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)

            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()

        _tokenized = tokenizer(cls_list[cls_id], return_tensors="pt", add_special_tokens=False)
        text_tokens = _tokenized["input_ids"].squeeze(0)

        return tokens, attention_masks, vision_patches, vision_patch_indices, labels, text_tokens, cur_target_mask, cur_target_boxes


    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']
            image = Image.open(str(image_path)).convert('RGB')
            image_array = get_transform()(image)
            seg_array = np.load(seg_path)

            # image_array = np.load(image_path) #1*32*256*256, normalized
            image_array_np = image_array.numpy()
            # cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])
            if image_array_np.shape[1:] != seg_array.shape:
                print(image_path)
            try:
                item = {
                    'image': image_array[:, :, :, np.newaxis],
                    'seg': seg_array[np.newaxis, :][:, :, :, np.newaxis],
                }

                it = self.transform(item)

                image = get_transform2()(it['image'].squeeze(-1))
                seg = it['seg'].squeeze(-1).squeeze(1)  # 1*D*H*W
                # seg = F.one_hot(seg.long(), num_classes=4)
                # img_patches = preprocess_image(image, patch_size=self.Patch_size)
                # cls_id_list = list(torch.nonzero(seg.squeeze().sum(dim=[0, 1]) > 0))[1:]
                seg = F.one_hot(seg.long(), num_classes=4)

                target_masks = torch.zeros(3)
                target_boxes = torch.zeros(3, 4)
                for t in range(3):
                    seg_O = seg.permute(0, 3, 1, 2).squeeze()[t + 1, :, :].unsqueeze(0)
                    mask_O = seg_O.numpy().squeeze()
                    if not np.any(mask_O):
                        continue  # 跳过无像素的类别
                    rows, cols = np.where(mask_O)
                    image_w = mask_O.shape[0]
                    image_h = mask_O.shape[1]
                    x_min, x_max = cols.min() / image_w, cols.max() / image_w
                    y_min, y_max = rows.min() / image_h, rows.max() / image_h

                    target_masks[t] = 1
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    target_boxes[t][0] = float(center_x)
                    target_boxes[t][1] = float(center_y)
                    target_boxes[t][2] = float(width)
                    target_boxes[t][3] = float(height)


                img_patches = preprocess_image(image, patch_size=self.Patch_size)
                cls_id = random.randint(0, 2)
                seg = seg.permute(0, 3, 1, 2).squeeze()[cls_id + 1, :, :].unsqueeze(0)
                # seg = seg.permute(0, 3, 1, 2).squeeze()

                tokens, attention_masks, vision_patches, vision_patch_indices, labels, text_tokens, target_masks, target_boxes = self.prepare_inputs_img(
                    img_patches, self.tokenizer, cls_id, target_masks, target_boxes)

                # detection_boxes_bbox = []
                # detection_boxes_class = []
                # # for cls in [1, 2, 3]:
                #     # 获取当前类别的掩码
                # mask = seg.numpy().squeeze()
                # if not np.any(mask):
                #     continue  # 跳过无像素的类别
                #
                # # 提取像素坐标
                # rows, cols = np.where(mask)
                # image_w = mask.shape[0]
                # image_h = mask.shape[1]
                # x_min, x_max = cols.min()/image_w, cols.max()/image_w
                # y_min, y_max = rows.min()/image_h, rows.max()/image_h
                #
                #
                # # 添加类别和框坐标 (x_min, y_min, x_max, y_max)
                # detection_boxes_class.append(cls_id)
                # detection_boxes_bbox.append([float(x_min), float(y_min), float(x_max), float(y_max)])


                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patches": vision_patches,
                    "vision_patch_indices": vision_patch_indices,
                    "labels": labels,
                    'seg': seg,
                    'image': image,
                    'text_tokens': text_tokens,
                    'cls_id': cls_id,
                    'image_path': image_path,
                    'seg_button': False,
                    # 'detection_button': True,
                    'detection_boxes_class': target_masks.unsqueeze(0),
                    'detection_boxes_bbox': target_boxes.unsqueeze(0)
                }

                # ret = {
                #     'image': image,
                #     'input_id': input_id,
                #     'label': label,
                #     'seg': seg,
                #     'attention_mask': attention_mask,
                #     'question': question,
                #     'answer': answer,
                #     'question_type': "seg",
                # }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class UniDatasets_tsvlge(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names=['FCH', 'SAX', 'LGE'], prompt_mode='classification',
                 n_class=7, multilabel=False, use_seg=True, use_numpy = False, use_det = False, Multi_center = 'KM',
                 abnormal_name= 'LVEDD'):
        super(UniDatasets_tsvlge, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.Multi_center = Multi_center
        print()
        if Multi_center == 'KM':
            with open(args.all_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_KM_columns_as_lists
            self.data_all_root = {
                'image': args.data_root,
                'LV_mask': args.seg_Lv_root,
                'RV_mask': args.seg_Rv_root,
                'MYO_mask': args.seg_MYO_root,
                'DET_mask': args.det_km_root,
            }
            self.weights_df = calculate_cardiac_weights(KM_excel_path)
        elif Multi_center == 'SCS':
            with open(args.scs_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_SCS_columns_as_lists
            self.data_all_root = {
                'image': args.scs_root,
                'LV_mask': args.seg_scs_Lv_root,
                'RV_mask': args.seg_scs_Rv_root,
                'MYO_mask': args.seg_scs_MYO_root,
                'DET_mask': args.det_scs_root,
            }
            self.weights_df = calculate_cardiac_weights(SCS_excel_path)
        elif Multi_center == 'CD':
            with open(args.cd_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_CD_columns_as_lists
            self.data_all_root = {
                'image': args.cd_root,
                'LV_mask': args.seg_cd_Lv_root,
                'RV_mask': args.seg_cd_Rv_root,
                'MYO_mask': args.seg_cd_MYO_root,
                'DET_mask': args.det_cd_root,
            }
            self.weights_df = calculate_cardiac_weights(CD_excel_path)
        elif Multi_center == 'NCSD':
            with open(args.location_data_path3D, 'r') as file:
                self.json_file = json.load(file)
            self.data_all_root = {
                'image': args.data_root,
                'LV_mask': args.seg_cd_Lv_root,#!
                'RV_mask': args.seg_cd_Rv_root,
                'MYO_mask': args.seg_cd_MYO_root,
            }
        self.data_list = self.json_file[mode]
        self.dataset = []
        self.dataset_name = dataset_names
        self.n_class = n_class
        print(f'Number class: ' + str(self.n_class))
        self.multilabel = multilabel
        self.use_seg = use_seg
        self.use_numpy = use_numpy
        self.use_det = use_det
        for dataset_name in dataset_names:
            if dataset_name == 'SAX':
                self.sax_dataset = SAXCineDataset_CMR_3DFilm_vst(self.args, self.data_all_root, self.tokenizer,
                                                                 self.json_file, self.G_columns_as_lists, self.mode,
                                                                 use_seg=self.use_seg, use_numpy = self.use_numpy, use_det =  self.use_det)
                self.dataset.append(self.sax_dataset)
            if dataset_name == 'FCH':
                self.fch_dataset = FCHCineDataset_CMR_2DFilm_vst(self.args, self.data_all_root, self.tokenizer,
                                                                 self.json_file, self.G_columns_as_lists,self.mode,
                                                                 use_numpy = self.use_numpy)  # 4CH
                self.dataset.append(self.fch_dataset)
            if dataset_name == 'LGE':
                self.lge_dataset = LGEDataset_CMR_3D_ST(self.args, self.data_all_root, self.tokenizer,
                                                        self.json_file, self.G_columns_as_lists,self.mode,
                                                        use_seg=self.use_seg, use_numpy = self.use_numpy, use_det = self.use_det)
                self.dataset.append(self.lge_dataset)

        self.max_position_embeddings = 4096
        self.prompt_mode = prompt_mode
        self.abnormal_name =abnormal_name
        if self.prompt_mode == 'classification':
            self.valid_idx = self.filter_class_label_numpy()

    def __len__(self):
        if self.prompt_mode == 'classification':
            return len(self.valid_idx)
        else:
            return len(self.dataset[0])


    def filter_class_label_numpy(self):
        valid_idx = []
        for data_index, data in enumerate(self.data_list):
            excel_id =  data['Text']['excel_id'] #G_columns_as_lists[]
            valid_class_label = []
            for class_n in CLASSES_CN:
                try:
                    if self.G_columns_as_lists[class_n][excel_id] == 1:
                        valid_class_label.append(class_n)
                except:
                    pass
            if len(valid_class_label) > 0:
                valid_idx.append(data_index)
                self.data_list[data_index]['classes'] = valid_class_label[0]
        print(CLASSES_CN)
        return valid_idx

    def prepare_inputs_img_text(self, input_img_tokens, input_img_patch_indices, input_text,
                                tokenizer, clinical_infor = '',
                                question_squeence_list = [3], question_id = 'Question_open', abnormal = ''):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]

        update_patch_indices = [cur_index + len(cur_patch_indices) - cur_patch_indices.count(
            NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                input_img_patch_indices]

        cur_patch_indices = cur_patch_indices + update_patch_indices  # include the whole <vision>...<vision>
        img_tokens = img_tokens + input_img_tokens  # all datasets should concat this

        img_tokens.append("/<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)


        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)

        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)

        if self.prompt_mode == 'caption':
            question = clinical_infor + random.choice(caption_prompt)
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'report':
            question = clinical_infor + random.choice(report_prompt)
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'classification':
            # question = additional_classification_prompt + '\nQuestion: ' + random.choice(classification_prompt) + '\nAnswer: '
            question = clinical_infor + additional_classification_prompt_larry + "<CLS>"
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'Question_open':
            if question_id == 'Question_open_no_2':
                QA_list = extract_content(input_text, 4)
            elif question_id == 'Question_open_no_3':
                QA_list = extract_content(input_text, 5)
            else:
                QA_list = extract_content(input_text, random.choice(question_squeence_list))

            try:
                if QA_list!= None:
                    QA = random.choice(QA_list)
                    Q = 'Question: '+ QA['question']
                    answer = 'Answer: '+QA['answer'] + end_token
                    few_shot = 0
                    if few_shot:
                        question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
                    else:
                        question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
            except:
                return None
        # elif question_id == 'Question_open_no_2':
        #     QA_list = extract_content(input_text, 4)
        #     try:
        #         if QA_list!= None:
        #             QA = random.choice(QA_list)
        #             Q = 'Question: '+ QA['question']
        #             answer = 'Answer: '+QA['answer'] + end_token
        #             few_shot = 0
        #             if few_shot:
        #                 question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
        #             else:
        #                 question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
        #     except:
        #         return None
        # elif question_id == 'Question_open_no_3':
        #     QA_list = extract_content(input_text, 5)
        #     try:
        #         if QA_list!= None:
        #             QA = random.choice(QA_list)
        #             Q = 'Question: '+ QA['question']
        #             answer = 'Answer: '+QA['answer'] + end_token
        #             few_shot = 0
        #             if few_shot:
        #                 question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
        #             else:
        #                 question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
        #     except:
        #         return None
        elif self.prompt_mode == 'Question_close':
            QA_list = extract_content_close(input_text, random.choice(question_squeence_list))
            try:
                if QA_list!= None:
                    QA = random.choice(QA_list)
                    Q = f"Question: {QA['question']} \n"
                    Q += "Options:"
                    for letter, content in QA['shuffled_options']:
                        Q+=f"  {letter}. {content}"
                    answer = f"Answer: {QA['randanswer_letter']}"  + end_token
                    question = clinical_infor + random.choice(question_close_prompt) + "\n" + Q
            except:
                return None

        elif self.prompt_mode == 'abnormal_close':
            question = r'CMR影像中的“{}”是否存在异常？'.format(abnormal)
            # question = clinical_infor + random.choice(report_prompt)
            answer = input_text
            answer = answer + end_token

        # img_tokens.append("<CLS>")
        # cur_patch_indices.append(NON_VISION_TOKEN)
        # if self.prompt_mode == 'caption':
        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))


        if self.mode == 'train':
            # 不能给classification任务添加answer token，不然会让模型学到answer text token和实际label的关系
            if self.prompt_mode == 'classification':
                # random.shuffle(answer)
                answer = ''+ end_token #str(answer)
            # print(answer)
            _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
            cur_tokens = _tokenized["input_ids"].squeeze(0)
            cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
            tokens.extend(cur_tokens)
            labels.extend(cur_tokens)
            attention_masks.extend(cur_attention_mask)
            vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()

        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patch_indices, labels, answer, question

    def __getitem__(self, idx):
        max_attempts = 10
        NON_VISION_TOKEN = -1
        for _ in range(max_attempts):
            try:
                if self.prompt_mode == 'classification':
                    idx = self.valid_idx[idx]
                    class_label = [self.data_list[idx]['classes']]
                    valid_class_label = class_label

                vision_tokens = []
                vision_patch_indices = []
                input_texts = []
                # sax_vision_patches = None
                # fch_vision_patches = None
                lge_vision_patches = None
                sax_vision_org = None
                fch_vision_org = None
                lge_vision_org = None
                vision_flag = False
                question_squeence_list= []
                for dataset_ind, dataset in enumerate(self.dataset):
                    # import pdb;pdb.set_trace()
                    if self.dataset_name[dataset_ind] == 'SAX':
                        try:
                            dataset_item = dataset[idx]
                            # sax_vision_patches = dataset_item['vision_patches']
                            sax_vision_org = dataset_item['org_image_list']
                            if sax_vision_org is None:
                                continue
                            else:
                                vision_sax_flag = True
                                question_squeence_list.append(1)
                        except:
                            sax_vision_org = None
                            continue
                    if self.dataset_name[dataset_ind] == 'FCH':
                        try:
                            dataset_item = dataset[idx]
                            # fch_vision_patches = dataset_item['vision_patches']
                            fch_vision_org = dataset_item['org_image']
                            if fch_vision_org is None:
                                continue
                            else:
                                vision_fch_flag = True
                                question_squeence_list.append(0)
                        except:
                            fch_vision_org = None
                            continue
                    if self.dataset_name[dataset_ind] == 'LGE':
                        try:
                            dataset_item = dataset[idx]
                            lge_vision_org = dataset_item['org_image']
                            if lge_vision_org is None:
                                continue
                            else:
                                vision_lge_flag = True
                                question_squeence_list.append(2)
                        except:
                            lge_vision_org = None
                            continue

                    vision_tokens.extend(dataset_item["input_ids"])
                    update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                        NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                            dataset_item['vision_patch_indices']]
                    vision_patch_indices.extend(update_patch_indices)
                    input_texts.append(dataset_item['text'])


                input_texts = '\n'.join(input_texts)
                # if self.use_numpy:
                balance_loss = 1

                clinical_infor = ''
                if self.Multi_center in ['KM','CD','SCS']:
                    clinical_infor = '临床信息: '
                    B = self.G_columns_as_lists['性别'][self.data_list[idx]['Text']['excel_id']]
                    clinical_infor += '性别: ' + str(B)
                    C = self.G_columns_as_lists['年龄'][self.data_list[idx]['Text']['excel_id']]
                    clinical_infor += ' | '+ '年龄: ' + str(C)
                    if self.Multi_center == 'KM':
                        A = self.G_columns_as_lists['临床诊断C'][self.data_list[idx]['Text']['excel_id']]
                        clinical_infor += ' | '+ '临床信息: '+str(A)
                    clinical_infor += '\n'


                if self.prompt_mode == 'caption':
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_texts, self.tokenizer, clinical_infor)
                    class_label = 'None'
                elif self.prompt_mode == 'report':
                    input_texts_org =  self.G_columns_as_lists['Trans_3'][self.data_list[idx]['Text']['excel_id']]
                    input_texts_json = json.loads(input_texts_org)
                    input_texts_4CH = "，".join(input_texts_json["1.心脏结构"])
                    cardiac_function = input_texts_json.get("2.心脏运动及功能") or input_texts_json.get("2.心脏功能")
                    input_texts_SAX = "，".join(cardiac_function)
                    input_texts_LGE = "，".join(input_texts_json["3.延迟强化LGE"])
                    input_texts_other = "，".join(input_texts_json["4.其他影像所见"])
                    input_texts = ''
                    if 'FCH' in self.dataset_name:
                        input_texts += ('心脏结构: ' + input_texts_4CH)
                    if 'SAX' in self.dataset_name:
                        input_texts += ('心脏运动及功能: ' + input_texts_SAX)
                    if 'LGE' in self.dataset_name:
                        input_texts += ('延迟强化LGE: ' + input_texts_LGE)

                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_texts, self.tokenizer, clinical_infor)
                    class_label = 'None'
                elif self.prompt_mode == 'Question_open' or self.prompt_mode == 'Question_close':
                    question_squeence_list = question_squeence_list*5 + [3]
                    question_id = 'Question_open'
                    if self.prompt_mode == 'Question_open':
                        if self.Multi_center == 'CD':
                            question_id = random.choice(['Question_open', 'Question_open_R1', 'Question_open', 'Question_open_R1', 'Question_open_no_2', 'Question_open_no_3'])
                            data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]
                        else:
                            question_id = random.choice(['Question_open', 'Question_open_R1'])
                            data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]
                    else:
                        question_id =random.choice(['Question_close', 'Question_close_R1'])
                        data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]

                    # data_str = G_columns_as_lists['Question_close'][self.data_list[idx]['Text']['excel_id']]
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, data_str, self.tokenizer, clinical_infor, question_squeence_list, question_id)
                    class_label = 'None'
                elif self.prompt_mode == 'abnormal_close':
                    input_texts_org = self.G_columns_as_lists['Trans_4'][self.data_list[idx]['Text']['excel_id']]
                    value = -1
                    for _ in range(5):
                        try:
                            if self.mode == 'train':
                                chinese_name = random.choice(list(name_mapping.keys()))
                            else:
                                chinese_name = self.abnormal_name
                            english_name = name_mapping[chinese_name]
                            pattern = r'\s*"{}",\s*"abnormal":\s*(true|false)'.format(chinese_name)
                            match = re.search(pattern, input_texts_org, re.IGNORECASE)  # 忽略大小写

                            if match:
                                abnormal_value = match.group(1).lower()  # 获取 true/false
                                if abnormal_value == 'true':
                                    value = 1
                                else:
                                    value = 0
                            else:
                                value = -1

                            if value != -1:
                                break
                        except:
                            pass



                    try:
                        onehot_label = torch.LongTensor([0] * 2)
                        if value == 1:
                            balance_loss = self.weights_df['weight_abnormal'][english_name]
                            onehot_label[1] = 1
                        else:
                            balance_loss = self.weights_df['weight_normal'][english_name]
                            onehot_label[0] = 1
                    except:
                        print(english_name)
                    input_text= str(value)
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_text, self.tokenizer, clinical_infor, abnormal=chinese_name)
                    class_label = onehot_label
                elif self.prompt_mode == 'classification':
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, '', self.tokenizer, clinical_infor)

                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patch_indices": patch_indices,
                    "labels": labels,
                    "class_label": class_label,
                    "balance_loss":balance_loss
                }

                if sax_vision_org is not None:
                    # ret['sax_vision_patches'] = sax_vision_patches
                    if len(sax_vision_org) == 1:
                        ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                    elif len(sax_vision_org) == 2:
                        ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                        ret['sax_vision_org_1'] = sax_vision_org[1].bfloat16()
                    elif len(sax_vision_org) == 3:
                        ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                        ret['sax_vision_org_1'] = sax_vision_org[1].bfloat16()
                        ret['sax_vision_org_2'] = sax_vision_org[2].bfloat16()
                    else:
                        pass



                if fch_vision_org is not None:
                    ret['fch_vision_org'] = fch_vision_org.bfloat16()

                # if fch_vision_patches is not None:
                #     ret['fch_vision_patches'] = fch_vision_patches
                if lge_vision_org is not None:
                    ret['lge_vision_org'] = lge_vision_org.bfloat16()



                if self.mode == 'test':
                    if question is None:
                        ret['question'] = ''
                    else:
                        ret['question'] = question
                    if answer is None:
                        ret['text'] = ''
                    else:
                        ret['text'] = answer
                        # print(answer)
                if self.prompt_mode == 'classification':
                    onehot_label = torch.LongTensor([0] * self.n_class)
                    for class_n in valid_class_label:
                        if self.n_class == 2:
                            class_ind = CLASSES_CN.index(class_n)
                            if class_ind == CLASSES_CN.index('Normal'):
                                onehot_label[0] = 1
                            else:
                                onehot_label[1] = 1
                        else:
                            class_ind = CLASSES_CN.index(class_n)
                            onehot_label[class_ind] = 1
                    if self.n_class == 2 and onehot_label[0] == 1 and onehot_label[1] == 1:
                        onehot_label[0] = 0
                    ret['class_label'] = onehot_label
                    ret['multilabel'] = self.multilabel
                return ret
            except Exception as e:
                print(f"Error in __getitem__ at index : {e}")
                if self.prompt_mode == 'classification':
                    idx = random.randint(0, len(self.valid_idx) - 1)
                else:
                    idx = random.randint(0, len(self.dataset[0]) - 1)

class UniDatasets_tsvlge_ALL(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names=['FCH', 'SAX', 'LGE'], prompt_mode='classification',
                 n_class=7, multilabel=False, use_seg=True, use_numpy = False, use_det = False, Multi_center = 'KM',
                 abnormal_name= 'LVEDD'):
        super(UniDatasets_tsvlge_ALL, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.Multi_center = Multi_center
        print()
        if Multi_center == 'KM':
            with open(args.all_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_KM_columns_as_lists
            self.data_all_root = {
                'image': args.data_root,
                'LV_mask': args.seg_Lv_root,
                'RV_mask': args.seg_Rv_root,
                'MYO_mask': args.seg_MYO_root,
                'DET_mask': args.det_km_root,
            }
            self.weights_df = calculate_cardiac_weights(KM_excel_path)
        elif Multi_center == 'SCS':
            with open(args.scs_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_SCS_columns_as_lists
            self.data_all_root = {
                'image': args.scs_root,
                'LV_mask': args.seg_scs_Lv_root,
                'RV_mask': args.seg_scs_Rv_root,
                'MYO_mask': args.seg_scs_MYO_root,
                'DET_mask': args.det_scs_root,
            }
            self.weights_df = calculate_cardiac_weights(SCS_excel_path)
        elif Multi_center == 'CD':
            with open(args.cd_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_CD_columns_as_lists
            self.data_all_root = {
                'image': args.cd_root,
                'LV_mask': args.seg_cd_Lv_root,
                'RV_mask': args.seg_cd_Rv_root,
                'MYO_mask': args.seg_cd_MYO_root,
                'DET_mask': args.det_cd_root,
            }
            self.weights_df = calculate_cardiac_weights(CD_excel_path)
        elif Multi_center == 'NCSD':
            with open(args.location_data_path3D, 'r') as file:
                self.json_file = json.load(file)
            self.data_all_root = {
                'image': args.data_root,
                'LV_mask': args.seg_cd_Lv_root,#!
                'RV_mask': args.seg_cd_Rv_root,
                'MYO_mask': args.seg_cd_MYO_root,
            }
        self.data_list = self.json_file[mode]
        self.dataset = []
        self.dataset_name = dataset_names
        self.n_class = n_class
        print(f'Number class: ' + str(self.n_class))
        self.multilabel = multilabel
        self.use_seg = use_seg
        self.use_numpy = use_numpy
        self.use_det = use_det
        for dataset_name in dataset_names:
            if dataset_name == 'SAX':
                self.sax_dataset = SAXCineDataset_CMR_3DFilm_vst_ALL(self.args, self.data_all_root, self.tokenizer,
                                                                 self.json_file, self.G_columns_as_lists, self.mode,
                                                                 use_seg=self.use_seg, use_numpy = self.use_numpy, use_det =  self.use_det)
                self.dataset.append(self.sax_dataset)
            if dataset_name == 'FCH':
                self.fch_dataset = FCHCineDataset_CMR_2DFilm_vst(self.args, self.data_all_root, self.tokenizer,
                                                                 self.json_file, self.G_columns_as_lists,self.mode,
                                                                 use_numpy = self.use_numpy)  # 4CH
                self.dataset.append(self.fch_dataset)
            if dataset_name == 'LGE':
                self.lge_dataset = LGEDataset_CMR_3D_ST(self.args, self.data_all_root, self.tokenizer,
                                                        self.json_file, self.G_columns_as_lists,self.mode,
                                                        use_seg=self.use_seg, use_numpy = self.use_numpy, use_det = self.use_det)
                self.dataset.append(self.lge_dataset)

        self.max_position_embeddings = 4096
        self.prompt_mode = prompt_mode
        self.abnormal_name =abnormal_name
        if self.prompt_mode == 'classification':
            self.valid_idx = self.filter_class_label_numpy()

    def __len__(self):
        if self.prompt_mode == 'classification':
            return len(self.valid_idx)
        else:
            return len(self.dataset[0])


    def filter_class_label_numpy(self):
        valid_idx = []
        for data_index, data in enumerate(self.data_list):
            excel_id =  data['Text']['excel_id'] #G_columns_as_lists[]
            valid_class_label = []
            for class_n in CLASSES_CN:
                try:
                    if self.G_columns_as_lists[class_n][excel_id] == 1:
                        valid_class_label.append(class_n)
                except:
                    pass
            if len(valid_class_label) > 0:
                valid_idx.append(data_index)
                self.data_list[data_index]['classes'] = valid_class_label[0]
        print(CLASSES_CN)
        return valid_idx

    def prepare_inputs_img_text(self, input_img_tokens, input_img_patch_indices, input_text,
                                tokenizer, clinical_infor = '',
                                question_squeence_list = [3], question_id = 'Question_open', abnormal = ''):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]

        update_patch_indices = [cur_index + len(cur_patch_indices) - cur_patch_indices.count(
            NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                input_img_patch_indices]

        cur_patch_indices = cur_patch_indices + update_patch_indices  # include the whole <vision>...<vision>
        img_tokens = img_tokens + input_img_tokens  # all datasets should concat this

        img_tokens.append("/<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)


        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)

        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)

        if self.prompt_mode == 'caption':
            question = clinical_infor + random.choice(caption_prompt)
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'report':
            question = clinical_infor + random.choice(report_prompt)
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'classification':
            # question = additional_classification_prompt + '\nQuestion: ' + random.choice(classification_prompt) + '\nAnswer: '
            question = clinical_infor + additional_classification_prompt_larry + "<CLS>"
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'Question_open':
            if question_id == 'Question_open_no_2':
                QA_list = extract_content(input_text, 4)
            elif question_id == 'Question_open_no_3':
                QA_list = extract_content(input_text, 5)
            else:
                QA_list = extract_content(input_text, random.choice(question_squeence_list))

            try:
                if QA_list!= None:
                    QA = random.choice(QA_list)
                    Q = 'Question: '+ QA['question']
                    answer = 'Answer: '+QA['answer'] + end_token
                    few_shot = 0
                    if few_shot:
                        question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
                    else:
                        question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
            except:
                return None
        # elif question_id == 'Question_open_no_2':
        #     QA_list = extract_content(input_text, 4)
        #     try:
        #         if QA_list!= None:
        #             QA = random.choice(QA_list)
        #             Q = 'Question: '+ QA['question']
        #             answer = 'Answer: '+QA['answer'] + end_token
        #             few_shot = 0
        #             if few_shot:
        #                 question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
        #             else:
        #                 question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
        #     except:
        #         return None
        # elif question_id == 'Question_open_no_3':
        #     QA_list = extract_content(input_text, 5)
        #     try:
        #         if QA_list!= None:
        #             QA = random.choice(QA_list)
        #             Q = 'Question: '+ QA['question']
        #             answer = 'Answer: '+QA['answer'] + end_token
        #             few_shot = 0
        #             if few_shot:
        #                 question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
        #             else:
        #                 question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
        #     except:
        #         return None
        elif self.prompt_mode == 'Question_close':
            QA_list = extract_content_close(input_text, random.choice(question_squeence_list))
            try:
                if QA_list!= None:
                    QA = random.choice(QA_list)
                    Q = f"Question: {QA['question']} \n"
                    Q += "Options:"
                    for letter, content in QA['shuffled_options']:
                        Q+=f"  {letter}. {content}"
                    answer = f"Answer: {QA['randanswer_letter']}"  + end_token
                    question = clinical_infor + random.choice(question_close_prompt) + "\n" + Q
            except:
                return None

        elif self.prompt_mode == 'abnormal_close':
            question = r'CMR影像中的“{}”是否存在异常？'.format(abnormal)
            # question = clinical_infor + random.choice(report_prompt)
            answer = input_text
            answer = answer + end_token

        # img_tokens.append("<CLS>")
        # cur_patch_indices.append(NON_VISION_TOKEN)
        # if self.prompt_mode == 'caption':
        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))


        if self.mode == 'train':
            # 不能给classification任务添加answer token，不然会让模型学到answer text token和实际label的关系
            if self.prompt_mode == 'classification':
                # random.shuffle(answer)
                answer = ''+ end_token #str(answer)
            # print(answer)
            _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
            cur_tokens = _tokenized["input_ids"].squeeze(0)
            cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
            tokens.extend(cur_tokens)
            labels.extend(cur_tokens)
            attention_masks.extend(cur_attention_mask)
            vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()

        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patch_indices, labels, answer, question

    def __getitem__(self, idx):
        max_attempts = 10
        NON_VISION_TOKEN = -1
        for _ in range(max_attempts):
            try:
                if self.prompt_mode == 'classification':
                    idx = self.valid_idx[idx]
                    class_label = [self.data_list[idx]['classes']]
                    valid_class_label = class_label

                vision_tokens = []
                vision_patch_indices = []
                input_texts = []
                # sax_vision_patches = None
                # fch_vision_patches = None
                lge_vision_patches = None
                sax_vision_org = None
                fch_vision_org = None
                lge_vision_org = None
                vision_flag = False
                question_squeence_list= []
                for dataset_ind, dataset in enumerate(self.dataset):
                    # import pdb;pdb.set_trace()
                    if self.dataset_name[dataset_ind] == 'SAX':
                        try:
                            dataset_item = dataset[idx]
                            # sax_vision_patches = dataset_item['vision_patches']
                            sax_vision_org = dataset_item['org_image_list']
                            if sax_vision_org is None:
                                continue
                            else:
                                vision_sax_flag = True
                                question_squeence_list.append(1)
                        except:
                            sax_vision_org = None
                            continue
                    if self.dataset_name[dataset_ind] == 'FCH':
                        try:
                            dataset_item = dataset[idx]
                            # fch_vision_patches = dataset_item['vision_patches']
                            fch_vision_org = dataset_item['org_image']
                            if fch_vision_org is None:
                                continue
                            else:
                                vision_fch_flag = True
                                question_squeence_list.append(0)
                        except:
                            fch_vision_org = None
                            continue
                    if self.dataset_name[dataset_ind] == 'LGE':
                        try:
                            dataset_item = dataset[idx]
                            lge_vision_org = dataset_item['org_image']
                            if lge_vision_org is None:
                                continue
                            else:
                                vision_lge_flag = True
                                question_squeence_list.append(2)
                        except:
                            lge_vision_org = None
                            continue

                    vision_tokens.extend(dataset_item["input_ids"])
                    update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                        NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                            dataset_item['vision_patch_indices']]
                    vision_patch_indices.extend(update_patch_indices)
                    input_texts.append(dataset_item['text'])


                input_texts = '\n'.join(input_texts)
                # if self.use_numpy:
                balance_loss = 1

                clinical_infor = ''
                if self.Multi_center in ['KM','CD','SCS']:
                    clinical_infor = '临床信息: '
                    B = self.G_columns_as_lists['性别'][self.data_list[idx]['Text']['excel_id']]
                    clinical_infor += '性别: ' + str(B)
                    C = self.G_columns_as_lists['年龄'][self.data_list[idx]['Text']['excel_id']]
                    clinical_infor += ' | '+ '年龄: ' + str(C)
                    if self.Multi_center == 'KM':
                        A = self.G_columns_as_lists['临床诊断C'][self.data_list[idx]['Text']['excel_id']]
                        clinical_infor += ' | '+ '临床信息: '+str(A)
                    clinical_infor += '\n'


                if self.prompt_mode == 'caption':
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_texts, self.tokenizer, clinical_infor)
                    class_label = 'None'
                elif self.prompt_mode == 'report':
                    input_texts_org =  self.G_columns_as_lists['Trans_3'][self.data_list[idx]['Text']['excel_id']]
                    input_texts_json = json.loads(input_texts_org)
                    input_texts_4CH = "，".join(input_texts_json["1.心脏结构"])
                    cardiac_function = input_texts_json.get("2.心脏运动及功能") or input_texts_json.get("2.心脏功能")
                    input_texts_SAX = "，".join(cardiac_function)
                    input_texts_LGE = "，".join(input_texts_json["3.延迟强化LGE"])
                    input_texts_other = "，".join(input_texts_json["4.其他影像所见"])
                    input_texts = ''
                    if 'FCH' in self.dataset_name:
                        input_texts += ('心脏结构: ' + input_texts_4CH)
                    if 'SAX' in self.dataset_name:
                        input_texts += ('心脏运动及功能: ' + input_texts_SAX)
                    if 'LGE' in self.dataset_name:
                        input_texts += ('延迟强化LGE: ' + input_texts_LGE)

                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_texts, self.tokenizer, clinical_infor)
                    class_label = 'None'
                elif self.prompt_mode == 'Question_open' or self.prompt_mode == 'Question_close':
                    question_squeence_list = question_squeence_list*5 + [3]
                    question_id = 'Question_open'
                    if self.prompt_mode == 'Question_open':
                        if self.Multi_center == 'CD':
                            question_id = random.choice(['Question_open', 'Question_open_R1', 'Question_open', 'Question_open_R1', 'Question_open_no_2', 'Question_open_no_3'])
                            data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]
                        else:
                            question_id = random.choice(['Question_open', 'Question_open_R1'])
                            data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]
                    else:
                        question_id =random.choice(['Question_close', 'Question_close_R1'])
                        data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]

                    # data_str = G_columns_as_lists['Question_close'][self.data_list[idx]['Text']['excel_id']]
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, data_str, self.tokenizer, clinical_infor, question_squeence_list, question_id)
                    class_label = 'None'
                elif self.prompt_mode == 'abnormal_close':
                    input_texts_org = self.G_columns_as_lists['Trans_4'][self.data_list[idx]['Text']['excel_id']]
                    value = -1
                    for _ in range(5):
                        try:
                            if self.mode == 'train':
                                chinese_name = random.choice(list(name_mapping.keys()))
                            else:
                                chinese_name = self.abnormal_name
                            english_name = name_mapping[chinese_name]
                            pattern = r'\s*"{}",\s*"abnormal":\s*(true|false)'.format(chinese_name)
                            match = re.search(pattern, input_texts_org, re.IGNORECASE)  # 忽略大小写

                            if match:
                                abnormal_value = match.group(1).lower()  # 获取 true/false
                                if abnormal_value == 'true':
                                    value = 1
                                else:
                                    value = 0
                            else:
                                value = -1

                            if value != -1:
                                break
                        except:
                            pass



                    try:
                        onehot_label = torch.LongTensor([0] * 2)
                        if value == 1:
                            balance_loss = self.weights_df['weight_abnormal'][english_name]
                            onehot_label[1] = 1
                        else:
                            balance_loss = self.weights_df['weight_normal'][english_name]
                            onehot_label[0] = 1
                    except:
                        print(english_name)
                    input_text= str(value)
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_text, self.tokenizer, clinical_infor, abnormal=chinese_name)
                    class_label = onehot_label
                elif self.prompt_mode == 'classification':
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, '', self.tokenizer, clinical_infor)

                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patch_indices": patch_indices,
                    "labels": labels,
                    "class_label": class_label,
                    "balance_loss":balance_loss
                }

                if sax_vision_org is not None:
                    if sax_vision_org.size(2)<3 or sax_vision_org.size(0)<2:
                        if self.prompt_mode == 'classification':
                            idx = random.randint(0, len(self.valid_idx) - 1)
                        else:
                            idx = random.randint(0, len(self.dataset[0]) - 1)
                        continue
                    else:
                        ret['sax_vision_org_0'] = sax_vision_org.bfloat16()
                # if sax_vision_org is not None:
                #     # ret['sax_vision_patches'] = sax_vision_patches
                #     if len(sax_vision_org) == 1:
                #         ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                #     elif len(sax_vision_org) == 2:
                #         ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                #         ret['sax_vision_org_1'] = sax_vision_org[1].bfloat16()
                #     elif len(sax_vision_org) == 3:
                #         ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                #         ret['sax_vision_org_1'] = sax_vision_org[1].bfloat16()
                #         ret['sax_vision_org_2'] = sax_vision_org[2].bfloat16()
                #     else:
                #         pass



                if fch_vision_org is not None:
                    ret['fch_vision_org'] = fch_vision_org.bfloat16()

                # if fch_vision_patches is not None:
                #     ret['fch_vision_patches'] = fch_vision_patches
                if lge_vision_org is not None:
                    ret['lge_vision_org'] = lge_vision_org.bfloat16()



                if self.mode == 'test':
                    if question is None:
                        ret['question'] = ''
                    else:
                        ret['question'] = question
                    if answer is None:
                        ret['text'] = ''
                    else:
                        ret['text'] = answer
                        # print(answer)
                if self.prompt_mode == 'classification':
                    onehot_label = torch.LongTensor([0] * self.n_class)
                    for class_n in valid_class_label:
                        if self.n_class == 2:
                            class_ind = CLASSES_CN.index(class_n)
                            if class_ind == CLASSES_CN.index('Normal'):
                                onehot_label[0] = 1
                            else:
                                onehot_label[1] = 1
                        else:
                            class_ind = CLASSES_CN.index(class_n)
                            onehot_label[class_ind] = 1
                    if self.n_class == 2 and onehot_label[0] == 1 and onehot_label[1] == 1:
                        onehot_label[0] = 0
                    ret['class_label'] = onehot_label
                    ret['multilabel'] = self.multilabel
                return ret
            except Exception as e:
                print(f"Error in __getitem__ at index : {e}")
                if self.prompt_mode == 'classification':
                    idx = random.randint(0, len(self.valid_idx) - 1)
                else:
                    idx = random.randint(0, len(self.dataset[0]) - 1)

class UniDatasets_tsvlge_ALL2(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names=['FCH', 'SAX', 'LGE'], prompt_mode='classification',
                 n_class=7, multilabel=False, use_seg=True, use_numpy = False, use_det = False, Multi_center = 'KM',
                 abnormal_name= 'LVEDD'):
        super(UniDatasets_tsvlge_ALL2, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.Multi_center = Multi_center
        print()
        if Multi_center == 'KM':
            with open(args.all_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_KM_columns_as_lists
            self.data_all_root = {
                'image': args.data_root,
                'LV_mask': args.seg_Lv_root,
                'RV_mask': args.seg_Rv_root,
                'MYO_mask': args.seg_MYO_root,
                'DET_mask': args.det_km_root,
            }
            self.weights_df = calculate_cardiac_weights(KM_excel_path)
        elif Multi_center == 'SCS':
            with open(args.scs_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_SCS_columns_as_lists
            self.data_all_root = {
                'image': args.scs_root,
                'LV_mask': args.seg_scs_Lv_root,
                'RV_mask': args.seg_scs_Rv_root,
                'MYO_mask': args.seg_scs_MYO_root,
                'DET_mask': args.det_scs_root,
            }
            self.weights_df = calculate_cardiac_weights(SCS_excel_path)
        elif Multi_center == 'CD':
            with open(args.cd_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_CD_columns_as_lists
            self.data_all_root = {
                'image': args.cd_root,
                'LV_mask': args.seg_cd_Lv_root,
                'RV_mask': args.seg_cd_Rv_root,
                'MYO_mask': args.seg_cd_MYO_root,
                'DET_mask': args.det_cd_root,
            }
            self.weights_df = calculate_cardiac_weights(CD_excel_path)
        elif Multi_center == 'NCSD':
            with open(args.location_data_path3D, 'r') as file:
                self.json_file = json.load(file)
            self.data_all_root = {
                'image': args.data_root,
                'LV_mask': args.seg_cd_Lv_root,#!
                'RV_mask': args.seg_cd_Rv_root,
                'MYO_mask': args.seg_cd_MYO_root,
            }
        self.data_list = self.json_file[mode]
        self.dataset = []
        self.dataset_name = dataset_names
        self.n_class = n_class
        print(f'Number class: ' + str(self.n_class))
        self.multilabel = multilabel
        self.use_seg = use_seg
        self.use_numpy = use_numpy
        self.use_det = use_det
        for dataset_name in dataset_names:
            if dataset_name == 'SAX':
                self.sax_dataset = SAXCineDataset_CMR_3DFilm_vst_ALL(self.args, self.data_all_root, self.tokenizer,
                                                                 self.json_file, self.G_columns_as_lists, self.mode,
                                                                 use_seg=self.use_seg, use_numpy = self.use_numpy, use_det =  self.use_det)
                self.dataset.append(self.sax_dataset)
            if dataset_name == 'FCH':
                self.fch_dataset = FCHCineDataset_CMR_2DFilm_vst(self.args, self.data_all_root, self.tokenizer,
                                                                 self.json_file, self.G_columns_as_lists,self.mode,
                                                                 use_numpy = self.use_numpy)  # 4CH
                self.dataset.append(self.fch_dataset)
            if dataset_name == 'LGE':
                self.lge_dataset = LGEDataset_CMR_3D_vst(self.args, self.data_all_root, self.tokenizer,
                                                        self.json_file, self.G_columns_as_lists,self.mode,
                                                        use_seg=self.use_seg, use_numpy = self.use_numpy, use_det = self.use_det)
                self.dataset.append(self.lge_dataset)

        self.max_position_embeddings = 4096
        self.prompt_mode = prompt_mode
        self.abnormal_name =abnormal_name
        if self.prompt_mode == 'classification':
            self.valid_idx = self.filter_class_label_numpy()
        self.filter_class_label_numpy2()
        if self.prompt_mode == 'mace':
            self._preprocess_survival_data()
        # self.collect_center_abnormal_class_stats()

    def __len__(self):
        if self.prompt_mode == 'classification':
            return len(self.valid_idx)
        else:
            return len(self.dataset[0])

    def _preprocess_survival_data(self):
        """预处理生存分析数据，统一处理所有数据后再分训练/测试集"""
        # 合并训练集和测试集数据
        all_data = self.json_file['train'] + self.json_file['test']

        # 1. 收集所有有效随访时间
        valid_times = []
        for item in all_data:
            mace_time = item['Text'].get('mace_time')
            if not pd.isna(mace_time) and mace_time is not None:
                valid_times.append(mace_time)

        if len(valid_times) == 0:
            raise ValueError("没有有效的随访时间数据可用于生存分析")

        # 2. 计算全局时间分桶（基于所有数据）
        self.global_time_bins = np.quantile(valid_times, np.linspace(0, 1, 10))
        self.global_time_bins = np.unique(self.global_time_bins)

        # 处理特殊情况（如所有时间相同）
        if len(self.global_time_bins) < 2:
            max_time = max(valid_times)
            self.global_time_bins = np.linspace(0, max_time, 10)

        # 3. 计算全局中位数用于填充NaN
        self.global_median_time = np.median(valid_times)

        print(f"全局时间分桶: {self.global_time_bins}")
        print(f"全局中位数时间: {self.global_median_time}")

        # 4. 为当前模式（train/test）创建标签
        self._create_survival_labels_for_current_mode()

    def _create_survival_labels_for_current_mode(self):
        """为当前数据集模式（train/test）创建生存标签"""
        self.survival_labels = []
        self.valid_indices = []

        event_count = 0
        total_count = len(self.data_list)

        for idx, item in enumerate(self.data_list):
            time = item['Text'].get('mace_time')
            event = item['Text'].get('mace_cls', 0)  # 默认为0（未发生事件）

            # 填充NaN值
            if pd.isna(time) or time is None:
                time = self.global_median_time

            # 创建时间区间标签
            bin_labels = []
            for bin_time in self.global_time_bins:
                if time >= bin_time:
                    bin_labels.append(1.0)  # 生存到该时间
                elif event == 1 and time < bin_time:
                    bin_labels.append(0.0)  # 在该时间前发生事件
                    event_count += 1
                else:
                    bin_labels.append(1.0)  # 删失数据

            self.survival_labels.append(bin_labels)
            self.valid_indices.append(idx)

        self.survival_labels = np.array(self.survival_labels)

        print(f"{self.mode}集 - 总样本数: {total_count} "
              f"事件数: {event_count} "
              f"事件率: {event_count / total_count:.2%}")
    def filter_class_label_numpy2(self):
        valid_idx = []
        for data_index, data in enumerate(self.data_list):
            excel_id =  data['Text']['excel_id'] #G_columns_as_lists[]
            valid_class_label = []
            for class_n in CLASSES_CN_4:
                try:
                    if self.G_columns_as_lists[class_n][excel_id] == 1:
                        valid_class_label.append(class_n)
                except:
                    pass
            if len(valid_class_label) > 0:
                valid_idx.append(data_index)
                self.data_list[data_index]['classes2'] = valid_class_label[-1]
        print(CLASSES_CN)
        return valid_idx

    def filter_class_label_numpy(self):
        valid_idx = []
        for data_index, data in enumerate(self.data_list):
            excel_id =  data['Text']['excel_id'] #G_columns_as_lists[]
            valid_class_label = []
            for class_n in CLASSES_CN:
                try:
                    if self.G_columns_as_lists[class_n][excel_id] == 1:
                        valid_class_label.append(class_n)
                except:
                    pass
            if len(valid_class_label) > 0:
                valid_idx.append(data_index)
                self.data_list[data_index]['classes'] = valid_class_label[0]
        print(CLASSES_CN)
        return valid_idx

    def prepare_inputs_img_text(self, input_img_tokens, input_img_patch_indices, input_text,
                                tokenizer, clinical_infor = '',
                                question_squeence_list = [3], question_id = 'Question_open', abnormal = ''):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]

        update_patch_indices = [cur_index + len(cur_patch_indices) - cur_patch_indices.count(
            NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                input_img_patch_indices]

        cur_patch_indices = cur_patch_indices + update_patch_indices  # include the whole <vision>...<vision>
        img_tokens = img_tokens + input_img_tokens  # all datasets should concat this

        img_tokens.append("/<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)


        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)

        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)

        if self.prompt_mode == 'caption':
            question = clinical_infor + random.choice(caption_prompt)
            answer = input_text + end_token
        elif self.prompt_mode == 'mace':
            question = "根据患者的心脏MRI影像和临床信息，预测MACE事件发生的风险和时间。"
            answer = input_text + end_token
        elif self.prompt_mode == 'report':
            question = clinical_infor + random.choice(report_prompt)
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'classification':
            # question = additional_classification_prompt + '\nQuestion: ' + random.choice(classification_prompt) + '\nAnswer: '
            question = clinical_infor + additional_classification_prompt_larry + "<CLS>"
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'Question_open':
            if question_id == 'Question_open_no_2':
                QA_list = extract_content(input_text, 4)
            elif question_id == 'Question_open_no_3':
                QA_list = extract_content(input_text, 5)
            else:
                QA_list = extract_content(input_text, random.choice(question_squeence_list))

            try:
                if QA_list!= None:
                    QA = random.choice(QA_list)
                    Q = 'Question: '+ QA['question']
                    answer = 'Answer: '+QA['answer'] + end_token
                    few_shot = 0
                    if few_shot:
                        question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
                    else:
                        question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
            except:
                return None

        elif self.prompt_mode == 'Question_close':
            QA_list = extract_content_close(input_text, random.choice(question_squeence_list))
            try:
                if QA_list!= None:
                    QA = random.choice(QA_list)
                    Q = f"Question: {QA['question']} \n"
                    Q += "Options:"
                    for letter, content in QA['shuffled_options']:
                        Q+=f"  {letter}. {content}"
                    answer = f"Answer: {QA['randanswer_letter']}"  + end_token
                    question = clinical_infor + random.choice(question_close_prompt) + "\n" + Q
            except:
                return None

        elif self.prompt_mode == 'abnormal_close':
            question = r'CMR影像中的“{}”是否存在异常？'.format(abnormal)
            # question = clinical_infor + random.choice(report_prompt)
            answer = input_text
            answer = answer + end_token

        # img_tokens.append("<CLS>")
        # cur_patch_indices.append(NON_VISION_TOKEN)
        # if self.prompt_mode == 'caption':
        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))


        if self.mode == 'train':
            # 不能给classification任务添加answer token，不然会让模型学到answer text token和实际label的关系
            if self.prompt_mode == 'classification':
                # random.shuffle(answer)
                answer = ''+ end_token #str(answer)
            # print(answer)
            _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
            cur_tokens = _tokenized["input_ids"].squeeze(0)
            cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
            tokens.extend(cur_tokens)
            labels.extend(cur_tokens)
            attention_masks.extend(cur_attention_mask)
            vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()

        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patch_indices, labels, answer, question

    def collect_center_abnormal_class_stats(self):
        """
        Collect actual statistics of abnormalities and classifications for each center.
        Returns:
            tuple: (center_abnormal_counts, abnormal_class_counts)
        """
        # Initialize counters
        center_abnormal_counts = {}  # (center, abnormal) -> count
        abnormal_class_counts = {}  # (abnormal, class) -> count

        # Traverse the dataset
        for idx in range(len(self.data_list)):
            try:
                # Get center
                center = self.Multi_center

                # Get abnormalities from Trans_4 field
                abnormalities = set()
                excel_id = self.data_list[idx]['Text']['excel_id']
                input_texts_org = self.G_columns_as_lists['Trans_4'][excel_id]

                # Use the same matching logic as in __getitem__
                for chinese_name in name_mapping.keys():
                    pattern = r'\s*"{}",\s*"abnormal":\s*(true|false)'.format(chinese_name)
                    match = re.search(pattern, input_texts_org, re.IGNORECASE)
                    if match:
                        abnormal_value = match.group(1).lower()
                        if abnormal_value == 'true':
                            abnormalities.add(chinese_name)

                # Get classifications
                classifications = set()


                class_label = [self.data_list[idx]['classes2']]
                for class_n in class_label:
                    try:
                        class_ind = CLASSES_CN_4.index(class_n)
                        classifications.add(class_n)
                    except ValueError:
                        continue

                # Update counts
                for abnormal in abnormalities:
                    # Center -> Abnormal
                    key = (center, abnormal)
                    center_abnormal_counts[key] = center_abnormal_counts.get(key, 0) + 1

                    # Abnormal -> Class
                    for cls in classifications:
                        key = (abnormal, cls)
                        abnormal_class_counts[key] = abnormal_class_counts.get(key, 0) + 1

            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                continue

        return center_abnormal_counts, abnormal_class_counts

    def __getitem__(self, idx):
        max_attempts = 10
        NON_VISION_TOKEN = -1
        for _ in range(max_attempts):
            try:
                if self.prompt_mode == 'classification':
                    idx = self.valid_idx[idx]
                    class_label = [self.data_list[idx]['classes']]
                    valid_class_label = class_label

                vision_tokens = []
                vision_patch_indices = []
                input_texts = []
                # sax_vision_patches = None
                # fch_vision_patches = None
                lge_vision_patches = None
                sax_vision_org = None
                fch_vision_org = None
                lge_vision_org = None
                vision_flag = False
                question_squeence_list= []
                for dataset_ind, dataset in enumerate(self.dataset):
                    # import pdb;pdb.set_trace()
                    if self.dataset_name[dataset_ind] == 'SAX':
                        try:
                            dataset_item = dataset[idx]
                            # sax_vision_patches = dataset_item['vision_patches']
                            sax_vision_org = dataset_item['org_image_list']
                            if sax_vision_org is None:
                                continue
                            else:
                                vision_sax_flag = True
                                question_squeence_list.append(1)
                        except:
                            sax_vision_org = None
                            continue
                    if self.dataset_name[dataset_ind] == 'FCH':
                        try:
                            dataset_item = dataset[idx]
                            # fch_vision_patches = dataset_item['vision_patches']
                            fch_vision_org = dataset_item['org_image']
                            if fch_vision_org is None:
                                continue
                            else:
                                vision_fch_flag = True
                                question_squeence_list.append(0)
                        except:
                            fch_vision_org = None
                            continue
                    if self.dataset_name[dataset_ind] == 'LGE':
                        try:
                            dataset_item = dataset[idx]
                            lge_vision_org = dataset_item['org_image']
                            if lge_vision_org is None:
                                continue
                            else:
                                vision_lge_flag = True
                                question_squeence_list.append(2)
                        except:
                            lge_vision_org = None
                            continue

                    vision_tokens.extend(dataset_item["input_ids"])
                    update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                        NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                            dataset_item['vision_patch_indices']]
                    vision_patch_indices.extend(update_patch_indices)
                    input_texts.append(dataset_item['text'])


                input_texts = '\n'.join(input_texts)
                # if self.use_numpy:
                balance_loss = 1

                clinical_infor = ''
                if self.Multi_center in ['KM','CD','SCS']:
                    clinical_infor = '临床信息: '
                    B = self.G_columns_as_lists['性别'][self.data_list[idx]['Text']['excel_id']]
                    clinical_infor += '性别: ' + str(B)
                    C = self.G_columns_as_lists['年龄'][self.data_list[idx]['Text']['excel_id']]
                    clinical_infor += ' | '+ '年龄: ' + str(C)
                    if self.Multi_center == 'KM' and self.prompt_mode != 'caption':
                        A = self.G_columns_as_lists['临床诊断C'][self.data_list[idx]['Text']['excel_id']]
                        clinical_infor += ' | '+ '临床信息: '+str(A)
                    clinical_infor += '\n'


                if self.prompt_mode == 'caption':
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_texts, self.tokenizer, clinical_infor)
                    class_label = 'None'
                elif self.prompt_mode == 'report':
                    input_texts_org =  self.G_columns_as_lists['Trans_3'][self.data_list[idx]['Text']['excel_id']]
                    input_texts_json = json.loads(input_texts_org)
                    input_texts_4CH = "，".join(input_texts_json["1.心脏结构"])
                    cardiac_function = input_texts_json.get("2.心脏运动及功能") or input_texts_json.get("2.心脏功能")
                    input_texts_SAX = "，".join(cardiac_function)
                    input_texts_LGE = "，".join(input_texts_json["3.延迟强化LGE"])
                    input_texts_other = "，".join(input_texts_json["4.其他影像所见"])
                    input_texts = ''
                    if 'FCH' in self.dataset_name:
                        input_texts += ('心脏结构: ' + input_texts_4CH)
                    if 'SAX' in self.dataset_name:
                        input_texts += ('心脏运动及功能: ' + input_texts_SAX)
                    if 'LGE' in self.dataset_name:
                        input_texts += ('延迟强化LGE: ' + input_texts_LGE)

                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_texts, self.tokenizer, clinical_infor)
                    class_label = 'None'
                elif self.prompt_mode == 'Question_open' or self.prompt_mode == 'Question_close':
                    question_squeence_list = question_squeence_list*5 + [3]
                    question_id = 'Question_open'
                    if self.prompt_mode == 'Question_open':
                        if self.Multi_center == 'CD':
                            question_id = random.choice(['Question_open', 'Question_open_R1', 'Question_open', 'Question_open_R1', 'Question_open_no_2', 'Question_open_no_3'])
                            data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]
                        else:
                            question_id = random.choice(['Question_open', 'Question_open_R1'])
                            data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]
                    else:
                        question_id =random.choice(['Question_close', 'Question_close_R1'])
                        data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]

                    # data_str = G_columns_as_lists['Question_close'][self.data_list[idx]['Text']['excel_id']]
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, data_str, self.tokenizer, clinical_infor, question_squeence_list, question_id)
                    class_label = 'None'
                elif self.prompt_mode == 'abnormal_close':
                    input_texts_org = self.G_columns_as_lists['Trans_4'][self.data_list[idx]['Text']['excel_id']]
                    value = -1
                    for _ in range(5):
                        try:
                            if self.mode == 'train':
                                chinese_name = random.choice(list(name_mapping.keys()))
                            else:
                                chinese_name = self.abnormal_name
                            english_name = name_mapping[chinese_name]
                            pattern = r'\s*"{}",\s*"abnormal":\s*(true|false)'.format(chinese_name)
                            match = re.search(pattern, input_texts_org, re.IGNORECASE)  # 忽略大小写

                            if match:
                                abnormal_value = match.group(1).lower()  # 获取 true/false
                                if abnormal_value == 'true':
                                    value = 1
                                else:
                                    value = 0
                            else:
                                value = -1

                            if value != -1:
                                break
                        except:
                            pass



                    try:
                        onehot_label = torch.LongTensor([0] * 2)
                        if value == 1:
                            balance_loss = self.weights_df['weight_abnormal'][english_name]
                            onehot_label[1] = 1
                        else:
                            balance_loss = self.weights_df['weight_normal'][english_name]
                            onehot_label[0] = 1
                    except:
                        print(english_name)
                    input_text= str(value)
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_text, self.tokenizer, clinical_infor, abnormal=chinese_name)
                    class_label = onehot_label
                elif self.prompt_mode == 'classification':
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, '', self.tokenizer, clinical_infor)
                # ===== 生存分析任务处理 =====
                elif self.prompt_mode == 'mace':
                    # 获取MACE状态和随访时间
                    mace_status = self.data_list[idx]['Text'].get('mace_cls', 0)
                    survival_time = self.data_list[idx]['Text'].get('mace_time', 0)

                    # 处理NaN值
                    if pd.isna(survival_time):
                        survival_time = getattr(self, 'global_median_time', 0)

                    # 构建生存分析标签
                    survival_label = []
                    for bin_time in getattr(self, 'global_time_bins', [0]):
                        if survival_time >= bin_time:
                            survival_label.append(1.0)
                        elif mace_status == 1 and survival_time < bin_time:
                            survival_label.append(0.0)
                        else:
                            survival_label.append(1.0)

                    # 构建问题-答案对

                    answer = f"MACE状态: {'阳性' if mace_status == 1 else '阴性'}, 随访时间: {survival_time}天"

                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, answer, self.tokenizer, clinical_infor
                    )
                    class_label = 'None'

                    # 构建返回字典
                    # ret = {
                    #     "input_ids": tokens,
                    #     "attention_mask": attention_masks,
                    #     "vision_patch_indices": patch_indices,
                    #     "labels": labels,
                    #     "survival_label": torch.FloatTensor(survival_label),
                    #     "mace_status": mace_status,
                    #     "survival_time": survival_time,
                    #     "balance_loss": balance_loss
                    # }

                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patch_indices": patch_indices,
                    "labels": labels,
                    "class_label": class_label,
                    "balance_loss":balance_loss
                }

                if sax_vision_org is not None:
                    if sax_vision_org.size(2)<3 or sax_vision_org.size(0)<2:
                        if self.prompt_mode == 'classification':
                            idx = random.randint(0, len(self.valid_idx) - 1)
                        else:
                            idx = random.randint(0, len(self.dataset[0]) - 1)
                        continue
                    else:
                        ret['sax_vision_org_0'] = sax_vision_org.bfloat16()

                if fch_vision_org is not None:
                    ret['fch_vision_org'] = fch_vision_org.bfloat16()

                if lge_vision_org is not None:
                    ret['lge_vision_org'] = lge_vision_org.bfloat16()

                if self.mode == 'test':
                    if question is None:
                        ret['question'] = ''
                    else:
                        ret['question'] = question
                    if answer is None:
                        ret['text'] = ''
                    else:
                        ret['text'] = answer
                        # print(answer)
                if self.prompt_mode == 'classification':
                    onehot_label = torch.LongTensor([0] * self.n_class)
                    for class_n in valid_class_label:
                        if self.n_class == 2:
                            class_ind = CLASSES_CN.index(class_n)
                            if class_ind == CLASSES_CN.index('Normal'):
                                onehot_label[0] = 1
                            else:
                                onehot_label[1] = 1
                        else:
                            class_ind = CLASSES_CN.index(class_n)
                            onehot_label[class_ind] = 1
                    if self.n_class == 2 and onehot_label[0] == 1 and onehot_label[1] == 1:
                        onehot_label[0] = 0
                    ret['class_label'] = onehot_label
                    ret['multilabel'] = self.multilabel

                return ret
            except Exception as e:
                print(f"Error in __getitem__ at index : {e}")
                if self.prompt_mode == 'classification':
                    idx = random.randint(0, len(self.valid_idx) - 1)
                else:
                    idx = random.randint(0, len(self.dataset[0]) - 1)

class UniDatasets_tsvlge_ALL3(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names=['FCH', 'SAX', 'LGE'], prompt_mode='classification',
                 n_class=9, multilabel=False, use_seg=True, use_numpy = False, use_det = False, Multi_center = 'KM',
                 abnormal_name= 'LVEDD', Class_cn = CLASSES_CN_3):
        super(UniDatasets_tsvlge_ALL3, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.Multi_center = Multi_center
        self.CLASSES_CN = Class_cn
        print()
        if Multi_center == 'KM':
            with open(args.all_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_KM_columns_as_lists
            self.data_all_root = {
                'image': args.data_root,
                'LV_mask': args.seg_Lv_root,
                'RV_mask': args.seg_Rv_root,
                'MYO_mask': args.seg_MYO_root,
                'DET_mask': args.det_km_root,
            }
            self.weights_df = calculate_cardiac_weights(KM_excel_path)
        elif Multi_center == 'SCS':
            with open(args.scs_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_SCS_columns_as_lists
            self.data_all_root = {
                'image': args.scs_root,
                'LV_mask': args.seg_scs_Lv_root,
                'RV_mask': args.seg_scs_Rv_root,
                'MYO_mask': args.seg_scs_MYO_root,
                'DET_mask': args.det_scs_root,
            }
            self.weights_df = calculate_cardiac_weights(SCS_excel_path)
        elif Multi_center == 'CD':
            with open(args.cd_data_path, 'r') as file:
                self.json_file = json.load(file)
            self.G_columns_as_lists = G_CD_columns_as_lists
            self.data_all_root = {
                'image': args.cd_root,
                'LV_mask': args.seg_cd_Lv_root,
                'RV_mask': args.seg_cd_Rv_root,
                'MYO_mask': args.seg_cd_MYO_root,
                'DET_mask': args.det_cd_root,
            }
            self.weights_df = calculate_cardiac_weights(CD_excel_path)
        elif Multi_center == 'NCSD':
            with open(args.location_data_path3D, 'r') as file:
                self.json_file = json.load(file)
            self.data_all_root = {
                'image': args.data_root,
                'LV_mask': args.seg_cd_Lv_root,#!
                'RV_mask': args.seg_cd_Rv_root,
                'MYO_mask': args.seg_cd_MYO_root,
            }
        elif Multi_center == 'YA':
            with open(args.YA_data_path, 'r') as file:
                self.json_file = json.load(file)
                self.json_file['test'] = self.json_file['train']
            self.G_columns_as_lists = G_YA_columns_as_lists
            self.data_all_root = {
                'image': args.YA_root,
                'LV_mask': args.seg_YA_Lv_root,#!
                'RV_mask': args.seg_YA_Rv_root,
                'MYO_mask': args.seg_YA_MYO_root,
                'DET_mask': args.det_YA_root,
            }
        self.data_list = self.json_file[mode]
        self.dataset = []
        self.dataset_name = dataset_names
        self.n_class = n_class
        print(f'Number class: ' + str(self.n_class))
        self.multilabel = multilabel
        self.use_seg = use_seg
        self.use_numpy = use_numpy
        self.use_det = use_det
        for dataset_name in dataset_names:
            if dataset_name == 'SAX':
                self.sax_dataset = SAXCineDataset_CMR_3DFilm_vst_ALL(self.args, self.data_all_root, self.tokenizer,
                                                                 self.json_file, self.G_columns_as_lists, self.mode,
                                                                 use_seg=self.use_seg, use_numpy = self.use_numpy, use_det =  self.use_det)
                self.dataset.append(self.sax_dataset)
            if dataset_name == 'FCH':
                self.fch_dataset = FCHCineDataset_CMR_2DFilm_vst(self.args, self.data_all_root, self.tokenizer,
                                                                 self.json_file, self.G_columns_as_lists,self.mode,
                                                                 use_numpy = self.use_numpy)  # 4CH
                self.dataset.append(self.fch_dataset)
            if dataset_name == 'LGE':
                self.lge_dataset = LGEDataset_CMR_3D_vst(self.args, self.data_all_root, self.tokenizer,
                                                        self.json_file, self.G_columns_as_lists,self.mode,
                                                        use_seg=self.use_seg, use_numpy = self.use_numpy, use_det = self.use_det)
                self.dataset.append(self.lge_dataset)

        self.max_position_embeddings = 4096
        self.prompt_mode = prompt_mode
        self.abnormal_name =abnormal_name
        if self.prompt_mode == 'classification':
            self.valid_idx = self.filter_class_label_numpy()

    def __len__(self):
        if self.prompt_mode == 'classification':
            return len(self.valid_idx)
        else:
            return len(self.dataset[0])


    def filter_class_label_numpy(self):
        valid_idx = []
        for data_index, data in enumerate(self.data_list):
            excel_id =  data['Text']['excel_id'] #G_columns_as_lists[]
            valid_class_label = []
            for class_n in self.CLASSES_CN:
                try:
                    if self.G_columns_as_lists[class_n][excel_id] == 1:
                        valid_class_label.append(class_n)
                except:
                    pass
            if len(valid_class_label) > 0:
                valid_idx.append(data_index)
                self.data_list[data_index]['classes'] = valid_class_label[0]
        print(self.CLASSES_CN)
        return valid_idx

    def prepare_inputs_img_text(self, input_img_tokens, input_img_patch_indices, input_text,
                                tokenizer, clinical_infor = '',
                                question_squeence_list = [3], question_id = 'Question_open', abnormal = ''):
        end_token = tokenizer.eos_token

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]

        update_patch_indices = [cur_index + len(cur_patch_indices) - cur_patch_indices.count(
            NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                input_img_patch_indices]

        cur_patch_indices = cur_patch_indices + update_patch_indices  # include the whole <vision>...<vision>
        img_tokens = img_tokens + input_img_tokens  # all datasets should concat this

        img_tokens.append("/<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)


        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)

        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)

        if self.prompt_mode == 'caption':
            question = clinical_infor + random.choice(caption_prompt)
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'report':
            question = clinical_infor + random.choice(report_prompt)
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'classification':
            # question = additional_classification_prompt + '\nQuestion: ' + random.choice(classification_prompt) + '\nAnswer: '
            question = clinical_infor + additional_classification_prompt_larry + "<CLS>"
            answer = input_text  # text
            answer = answer + end_token
        elif self.prompt_mode == 'Question_open':
            if question_id == 'Question_open_no_2':
                QA_list = extract_content(input_text, 4)
            elif question_id == 'Question_open_no_3':
                QA_list = extract_content(input_text, 5)
            else:
                QA_list = extract_content(input_text, random.choice(question_squeence_list))

            try:
                if QA_list!= None:
                    QA = random.choice(QA_list)
                    Q = 'Question: '+ QA['question']
                    answer = 'Answer: '+QA['answer'] + end_token
                    few_shot = 0
                    if few_shot:
                        question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
                    else:
                        question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
            except:
                return None
        # elif question_id == 'Question_open_no_2':
        #     QA_list = extract_content(input_text, 4)
        #     try:
        #         if QA_list!= None:
        #             QA = random.choice(QA_list)
        #             Q = 'Question: '+ QA['question']
        #             answer = 'Answer: '+QA['answer'] + end_token
        #             few_shot = 0
        #             if few_shot:
        #                 question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
        #             else:
        #                 question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
        #     except:
        #         return None
        # elif question_id == 'Question_open_no_3':
        #     QA_list = extract_content(input_text, 5)
        #     try:
        #         if QA_list!= None:
        #             QA = random.choice(QA_list)
        #             Q = 'Question: '+ QA['question']
        #             answer = 'Answer: '+QA['answer'] + end_token
        #             few_shot = 0
        #             if few_shot:
        #                 question = clinical_infor + random.choice(question_open_prompt) + "(例如，Question: 左心房大小是否在正常范围内？ Answer: 是，左心房未见增大)" + "\n" + Q
        #             else:
        #                 question = clinical_infor + random.choice(question_open_prompt)  + "\n" + Q
        #     except:
        #         return None
        elif self.prompt_mode == 'Question_close':
            QA_list = extract_content_close(input_text, random.choice(question_squeence_list))
            try:
                if QA_list!= None:
                    QA = random.choice(QA_list)
                    Q = f"Question: {QA['question']} \n"
                    Q += "Options:"
                    for letter, content in QA['shuffled_options']:
                        Q+=f"  {letter}. {content}"
                    answer = f"Answer: {QA['randanswer_letter']}"  + end_token
                    question = clinical_infor + random.choice(question_close_prompt) + "\n" + Q
            except:
                return None

        elif self.prompt_mode == 'abnormal_close':
            question = r'CMR影像中的“{}”是否存在异常？'.format(abnormal)
            # question = clinical_infor + random.choice(report_prompt)
            answer = input_text
            answer = answer + end_token

        # img_tokens.append("<CLS>")
        # cur_patch_indices.append(NON_VISION_TOKEN)
        # if self.prompt_mode == 'caption':
        c_new = tokenizer.bos_token + f"{B_INST} {question.strip()} {E_INST}"
        _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
        cur_tokens = _tokenized["input_ids"].squeeze(0)
        cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens))
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))


        if self.mode == 'train':
            # 不能给classification任务添加answer token，不然会让模型学到answer text token和实际label的关系
            if self.prompt_mode == 'classification':
                # random.shuffle(answer)
                answer = ''+ end_token #str(answer)
            # print(answer)
            _tokenized = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
            cur_tokens = _tokenized["input_ids"].squeeze(0)
            cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
            tokens.extend(cur_tokens)
            labels.extend(cur_tokens)
            attention_masks.extend(cur_attention_mask)
            vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()

        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patch_indices, labels, answer, question

    def __getitem__(self, idx):
        max_attempts = 10
        NON_VISION_TOKEN = -1
        for _ in range(max_attempts):
            try:
                if self.prompt_mode == 'classification':
                    idx = self.valid_idx[idx]
                    class_label = [self.data_list[idx]['classes']]
                    valid_class_label = class_label

                vision_tokens = []
                vision_patch_indices = []
                input_texts = []
                # sax_vision_patches = None
                # fch_vision_patches = None
                lge_vision_patches = None
                sax_vision_org = None
                fch_vision_org = None
                lge_vision_org = None
                vision_flag = False
                question_squeence_list= []
                for dataset_ind, dataset in enumerate(self.dataset):
                    # import pdb;pdb.set_trace()
                    if self.dataset_name[dataset_ind] == 'SAX':
                        try:
                            dataset_item = dataset[idx]
                            # sax_vision_patches = dataset_item['vision_patches']
                            sax_vision_org = dataset_item['org_image_list']
                            if sax_vision_org is None:
                                continue
                            else:
                                vision_sax_flag = True
                                question_squeence_list.append(1)
                        except:
                            sax_vision_org = None
                            continue
                    if self.dataset_name[dataset_ind] == 'FCH':
                        try:
                            dataset_item = dataset[idx]
                            # fch_vision_patches = dataset_item['vision_patches']
                            fch_vision_org = dataset_item['org_image']
                            if fch_vision_org is None:
                                continue
                            else:
                                vision_fch_flag = True
                                question_squeence_list.append(0)
                        except:
                            fch_vision_org = None
                            continue
                    if self.dataset_name[dataset_ind] == 'LGE':
                        try:
                            dataset_item = dataset[idx]
                            lge_vision_org = dataset_item['org_image']
                            if lge_vision_org is None:
                                continue
                            else:
                                vision_lge_flag = True
                                question_squeence_list.append(2)
                        except:
                            lge_vision_org = None
                            continue

                    vision_tokens.extend(dataset_item["input_ids"])
                    update_patch_indices = [cur_index + len(vision_patch_indices) - vision_patch_indices.count(
                        NON_VISION_TOKEN) if cur_index != NON_VISION_TOKEN else NON_VISION_TOKEN for cur_index in
                                            dataset_item['vision_patch_indices']]
                    vision_patch_indices.extend(update_patch_indices)
                    input_texts.append(dataset_item['text'])


                input_texts = '\n'.join(input_texts)
                # if self.use_numpy:
                balance_loss = 1

                clinical_infor = ''
                if self.Multi_center in ['KM','CD','SCS','YA']:
                    clinical_infor = '临床信息: '
                    B = self.G_columns_as_lists['性别'][self.data_list[idx]['Text']['excel_id']]
                    clinical_infor += '性别: ' + str(B)
                    C = self.G_columns_as_lists['年龄'][self.data_list[idx]['Text']['excel_id']]
                    clinical_infor += ' | '+ '年龄: ' + str(C)
                    if self.Multi_center == 'KM':
                        A = self.G_columns_as_lists['临床诊断C'][self.data_list[idx]['Text']['excel_id']]
                        clinical_infor += ' | '+ '临床信息: '+str(A)
                    clinical_infor += '\n'


                if self.prompt_mode == 'caption':
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_texts, self.tokenizer, clinical_infor)
                    class_label = 'None'
                elif self.prompt_mode == 'report':
                    input_texts_org =  self.G_columns_as_lists['Trans_3'][self.data_list[idx]['Text']['excel_id']]
                    input_texts_json = json.loads(input_texts_org)
                    input_texts_4CH = "，".join(input_texts_json["1.心脏结构"])
                    cardiac_function = input_texts_json.get("2.心脏运动及功能") or input_texts_json.get("2.心脏功能")
                    input_texts_SAX = "，".join(cardiac_function)
                    input_texts_LGE = "，".join(input_texts_json["3.延迟强化LGE"])
                    input_texts_other = "，".join(input_texts_json["4.其他影像所见"])
                    input_texts = ''
                    if 'FCH' in self.dataset_name:
                        input_texts += ('心脏结构: ' + input_texts_4CH)
                    if 'SAX' in self.dataset_name:
                        input_texts += ('心脏运动及功能: ' + input_texts_SAX)
                    if 'LGE' in self.dataset_name:
                        input_texts += ('延迟强化LGE: ' + input_texts_LGE)

                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_texts, self.tokenizer, clinical_infor)
                    class_label = 'None'
                elif self.prompt_mode == 'Question_open' or self.prompt_mode == 'Question_close':
                    question_squeence_list = question_squeence_list*5 + [3]
                    question_id = 'Question_open'
                    if self.prompt_mode == 'Question_open':
                        if self.Multi_center == 'CD':
                            question_id = random.choice(['Question_open', 'Question_open_R1', 'Question_open', 'Question_open_R1', 'Question_open_no_2', 'Question_open_no_3'])
                            data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]
                        else:
                            question_id = random.choice(['Question_open', 'Question_open_R1'])
                            data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]
                    else:
                        question_id =random.choice(['Question_close', 'Question_close_R1'])
                        data_str = self.G_columns_as_lists[question_id][self.data_list[idx]['Text']['excel_id']]

                    # data_str = G_columns_as_lists['Question_close'][self.data_list[idx]['Text']['excel_id']]
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, data_str, self.tokenizer, clinical_infor, question_squeence_list, question_id)
                    class_label = 'None'
                elif self.prompt_mode == 'abnormal_close':
                    input_texts_org = self.G_columns_as_lists['Trans_4'][self.data_list[idx]['Text']['excel_id']]
                    value = -1
                    for _ in range(5):
                        try:
                            if self.mode == 'train':
                                chinese_name = random.choice(list(name_mapping.keys()))
                            else:
                                chinese_name = self.abnormal_name
                            english_name = name_mapping[chinese_name]
                            pattern = r'\s*"{}",\s*"abnormal":\s*(true|false)'.format(chinese_name)
                            match = re.search(pattern, input_texts_org, re.IGNORECASE)  # 忽略大小写

                            if match:
                                abnormal_value = match.group(1).lower()  # 获取 true/false
                                if abnormal_value == 'true':
                                    value = 1
                                else:
                                    value = 0
                            else:
                                value = -1

                            if value != -1:
                                break
                        except:
                            pass



                    try:
                        onehot_label = torch.LongTensor([0] * 2)
                        if value == 1:
                            balance_loss = self.weights_df['weight_abnormal'][english_name]
                            onehot_label[1] = 1
                        else:
                            balance_loss = self.weights_df['weight_normal'][english_name]
                            onehot_label[0] = 1
                    except:
                        print(english_name)
                    input_text= str(value)
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, input_text, self.tokenizer, clinical_infor, abnormal=chinese_name)
                    class_label = onehot_label
                elif self.prompt_mode == 'classification':
                    tokens, attention_masks, patch_indices, labels, answer, question = self.prepare_inputs_img_text(
                        vision_tokens, vision_patch_indices, '', self.tokenizer, clinical_infor)

                ret = {
                    "input_ids": tokens,
                    "attention_mask": attention_masks,
                    "vision_patch_indices": patch_indices,
                    "labels": labels,
                    "class_label": class_label,
                    "balance_loss":balance_loss
                }

                if sax_vision_org is not None:
                    if sax_vision_org.size(2)<3 or sax_vision_org.size(0)<2:
                        if self.prompt_mode == 'classification':
                            idx = random.randint(0, len(self.valid_idx) - 1)
                        else:
                            idx = random.randint(0, len(self.dataset[0]) - 1)
                        continue
                    else:
                        ret['sax_vision_org_0'] = sax_vision_org.bfloat16()
                # if sax_vision_org is not None:
                #     # ret['sax_vision_patches'] = sax_vision_patches
                #     if len(sax_vision_org) == 1:
                #         ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                #     elif len(sax_vision_org) == 2:
                #         ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                #         ret['sax_vision_org_1'] = sax_vision_org[1].bfloat16()
                #     elif len(sax_vision_org) == 3:
                #         ret['sax_vision_org_0'] = sax_vision_org[0].bfloat16()
                #         ret['sax_vision_org_1'] = sax_vision_org[1].bfloat16()
                #         ret['sax_vision_org_2'] = sax_vision_org[2].bfloat16()
                #     else:
                #         pass



                if fch_vision_org is not None:
                    ret['fch_vision_org'] = fch_vision_org.bfloat16()

                # if fch_vision_patches is not None:
                #     ret['fch_vision_patches'] = fch_vision_patches
                if lge_vision_org is not None:
                    ret['lge_vision_org'] = lge_vision_org.bfloat16()



                if self.mode == 'test':
                    if question is None:
                        ret['question'] = ''
                    else:
                        ret['question'] = question
                    if answer is None:
                        ret['text'] = ''
                    else:
                        ret['text'] = answer
                        # print(answer)
                if self.prompt_mode == 'classification':
                    onehot_label = torch.LongTensor([0] * self.n_class)
                    for class_n in valid_class_label:
                        if self.n_class == 2:
                            class_ind = self.CLASSES_CN.index(class_n)
                            if class_ind == self.CLASSES_CN.index('正常'):
                                onehot_label[0] = 1
                            else:
                                onehot_label[1] = 1
                        else:
                            class_ind = self.CLASSES_CN.index(class_n)
                            onehot_label[class_ind] = 1
                    if self.n_class == 2 and onehot_label[0] == 1 and onehot_label[1] == 1:
                        onehot_label[0] = 0
                    ret['class_label'] = onehot_label
                    ret['multilabel'] = self.multilabel
                return ret
            except Exception as e:
                print(f"Error in __getitem__ at index : {e}")
                if self.prompt_mode == 'classification':
                    idx = random.randint(0, len(self.valid_idx) - 1)
                else:
                    idx = random.randint(0, len(self.dataset[0]) - 1)

class AllDatasets_cls_Seg_vstlge_numpy_evalue(Dataset):
    def __init__(self, args, tokenizer, mode="test"):
        super(AllDatasets_cls_Seg_vstlge_numpy_evalue, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        self.dataset_cls_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH','SAX', 'LGE'],
                                       n_class = 7 , prompt_mode='classification', use_seg=True, use_numpy = False, Multi_center='SCS')
        self.dataset_cls_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
                                              n_class=7, prompt_mode='classification',
                                              use_seg=True, use_numpy=False, Multi_center='KM')

        # self.dataset_cls1 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls2 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX', 'LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='classification', use_seg=True, use_numpy=False)
        # self.dataset_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
        #                                prompt_mode='caption', use_seg=True)
        epoch = 1
        dataset_list = [
                           self.dataset_cls_km,
                           self.dataset_cls_scs,
                           # self.dataset_cls,
                           # self.dataset_cls,
                           # self.dataset_cls1,
                           # self.dataset_cls1,
                           # self.dataset_cls2,

                       ] * epoch
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class AllDatasets_cls_Seg_vstlge_numpy(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names = ['FCH','SAX', 'LGE']):
        super(AllDatasets_cls_Seg_vstlge_numpy, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.dataset_names = dataset_names
        # self.dataset_names = ['FCH', 'LGE']
        self.dataset_cls_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                       n_class = self.args.num_labels_D , prompt_mode='classification',
                                                  use_seg=True, use_numpy = False, Multi_center='SCS', use_det = True)
        self.dataset_cls_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                              n_class=self.args.num_labels_D, prompt_mode='classification',
                                              use_seg=True, use_numpy=False, Multi_center='KM', use_det = True)
        self.dataset_cls_cd = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
                                                 dataset_names=self.dataset_names,
                                                 n_class=self.args.num_labels_D, prompt_mode='classification',
                                                 use_seg=True, use_numpy=False, Multi_center='CD', use_det = True)

        # self.dataset_cls1 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls2 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX', 'LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='classification', use_seg=True, use_numpy=False)
        # self.dataset_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
        #                                prompt_mode='caption', use_seg=True)
        epoch = 1
        dataset_list = [
                           self.dataset_cls_km,
                           self.dataset_cls_scs,
                           self.dataset_cls_cd
                           # self.dataset_cls,
                           # self.dataset_cls,
                           # self.dataset_cls1,
                           # self.dataset_cls1,
                           # self.dataset_cls2,

                       ] * epoch
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_new(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names = ['FCH','SAX', 'LGE']):
        super(AllDatasets_cls_Seg_vstlge_numpy_new, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.dataset_names = dataset_names
        # self.dataset_names = ['FCH', 'LGE']
        self.dataset_cls_scs = UniDatasets_tsvlge_ALL(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                       n_class = self.args.num_labels_D , prompt_mode='classification',
                                                  use_seg=True, use_numpy = False, Multi_center='SCS', use_det = False)
        self.dataset_cls_km = UniDatasets_tsvlge_ALL(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                              n_class=self.args.num_labels_D, prompt_mode='classification',
                                              use_seg=True, use_numpy=False, Multi_center='KM', use_det = False)
        self.dataset_cls_cd = UniDatasets_tsvlge_ALL(self.args, self.tokenizer, self.mode,
                                                 dataset_names=self.dataset_names,
                                                 n_class=self.args.num_labels_D, prompt_mode='classification',
                                                 use_seg=True, use_numpy=False, Multi_center='CD', use_det = False)

        # self.dataset_cls1 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls2 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX', 'LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='classification', use_seg=True, use_numpy=False)
        # self.dataset_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
        #                                prompt_mode='caption', use_seg=True)
        epoch = 1
        dataset_list = [
                           self.dataset_cls_km,
                           self.dataset_cls_scs,
                           self.dataset_cls_cd
                           # self.dataset_cls,
                           # self.dataset_cls,
                           # self.dataset_cls1,
                           # self.dataset_cls1,
                           # self.dataset_cls2,

                       ] * epoch
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_new2(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names = ['FCH','SAX', 'LGE']):
        super(AllDatasets_cls_Seg_vstlge_numpy_new2, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.dataset_names = dataset_names
        # self.dataset_names = ['FCH', 'LGE']
        self.dataset_cls_scs = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                       n_class = self.args.num_labels_D , prompt_mode='classification',
                                                  use_seg=True, use_numpy = False, Multi_center='SCS', use_det = True)
        self.dataset_cls_km = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                              n_class=self.args.num_labels_D, prompt_mode='classification',
                                              use_seg=True, use_numpy=False, Multi_center='KM', use_det = True)
        self.dataset_cls_cd = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                                 dataset_names=self.dataset_names,
                                                 n_class=self.args.num_labels_D, prompt_mode='classification',
                                                 use_seg=True, use_numpy=False, Multi_center='CD', use_det = True)

        epoch = 1
        dataset_list = [
                           self.dataset_cls_km,
                           self.dataset_cls_scs,
                           self.dataset_cls_cd
                       ] * epoch
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_new3(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names = ['FCH','SAX', 'LGE'], class_cn = CLASSES_CN_3):
        super(AllDatasets_cls_Seg_vstlge_numpy_new3, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.dataset_names = dataset_names
        # self.dataset_names = ['FCH', 'LGE']
        self.dataset_cls_scs = UniDatasets_tsvlge_ALL3(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                       n_class = self.args.num_labels_D , prompt_mode='classification',
                                                  use_seg=True, use_numpy = False, Multi_center='SCS', use_det = True, Class_cn = class_cn)
        self.dataset_cls_km = UniDatasets_tsvlge_ALL3(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                              n_class=self.args.num_labels_D, prompt_mode='classification',
                                              use_seg=True, use_numpy=False, Multi_center='KM', use_det = True, Class_cn = class_cn)
        self.dataset_cls_cd = UniDatasets_tsvlge_ALL3(self.args, self.tokenizer, self.mode,
                                                 dataset_names=self.dataset_names,
                                                 n_class=self.args.num_labels_D, prompt_mode='classification',
                                                 use_seg=True, use_numpy=False, Multi_center='CD', use_det = True, Class_cn = class_cn)

        epoch = 1
        dataset_list = [
                           self.dataset_cls_km,
                           self.dataset_cls_scs,
                           self.dataset_cls_cd
                       ] * epoch
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_new3_ext(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names = ['FCH','SAX', 'LGE'], class_cn = CLASSES_CN_3):
        super(AllDatasets_cls_Seg_vstlge_numpy_new3_ext, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.dataset_names = dataset_names
        # self.dataset_names = ['FCH', 'LGE']
        self.dataset_cls_YA = UniDatasets_tsvlge_ALL3(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                       n_class = self.args.num_labels_D , prompt_mode='classification',
                                                  use_seg=True, use_numpy = False, Multi_center='YA', use_det = True, Class_cn = class_cn)

        dataset_list = [
                           self.dataset_cls_YA
                       ]
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_abnormal(Dataset):
    def __init__(self, args, tokenizer, mode="train", dataset_names = ['FCH','SAX', 'LGE'], abnormal_name= 'LVEDD'):
        super(AllDatasets_cls_Seg_vstlge_numpy_abnormal, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.dataset_names = dataset_names
        # self.dataset_names = ['FCH', 'LGE']
        self.dataset_cls_scs = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                       n_class = self.args.num_labels_D , prompt_mode='abnormal_close',
                                                  use_seg=True, use_numpy = False, Multi_center='SCS', use_det = True,
                                                  abnormal_name=abnormal_name)
        self.dataset_cls_km = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode, dataset_names=self.dataset_names,
                                              n_class=self.args.num_labels_D, prompt_mode='abnormal_close',
                                              use_seg=True, use_numpy=False, Multi_center='KM', use_det = True,
                                                    abnormal_name=abnormal_name)
        self.dataset_cls_cd = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                                 dataset_names=self.dataset_names,
                                                 n_class=self.args.num_labels_D, prompt_mode='abnormal_close',
                                                 use_seg=True, use_numpy=False, Multi_center='CD', use_det = True,
                                                 abnormal_name=abnormal_name)
        self.plot_center_abnormal_class_sankey()
        # self.dataset_cls1 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls2 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX', 'LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='classification', use_seg=True, use_numpy=False)
        # self.dataset_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
        #                                prompt_mode='caption', use_seg=True)
        epoch = 1
        dataset_list = [
                           self.dataset_cls_km,
                           self.dataset_cls_scs,
                           self.dataset_cls_cd
                           # self.dataset_cls,
                           # self.dataset_cls,
                           # self.dataset_cls1,
                           # self.dataset_cls1,
                           # self.dataset_cls2,

                       ] * epoch
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def plot_center_abnormal_class_sankey(self, save_path='sky.png', min_count=5):
        """
        绘制合并三个中心的精美Sankey图，展示中心→异常→分类的数据流

        Args:
            save_path (str, optional): 图片保存路径
            min_count (int): 显示连接的最小计数阈值
        """
        # 1. 收集并合并数据
        center_abnormal_counts_scs, abnormal_class_counts_scs = self.dataset_cls_scs.collect_center_abnormal_class_stats()
        center_abnormal_counts_km, abnormal_class_counts_km = self.dataset_cls_km.collect_center_abnormal_class_stats()
        center_abnormal_counts_cd, abnormal_class_counts_cd = self.dataset_cls_cd.collect_center_abnormal_class_stats()

        # 合并数据
        combined_center_abnormal = {}
        combined_abnormal_class = {}

        for counts_dict in [center_abnormal_counts_scs, center_abnormal_counts_km, center_abnormal_counts_cd]:
            for (center, abnormal), count in counts_dict.items():
                combined_center_abnormal[(center, abnormal)] = combined_center_abnormal.get((center, abnormal),
                                                                                            0) + count

        for counts_dict in [abnormal_class_counts_scs, abnormal_class_counts_km, abnormal_class_counts_cd]:
            for (abnormal, cls), count in counts_dict.items():
                combined_abnormal_class[(abnormal, cls)] = combined_abnormal_class.get((abnormal, cls), 0) + count

        # 2. 准备节点数据
        centers = ['SCS', 'KM', 'CD']  # 按特定顺序排列
        abnormalities = sorted(set([k[1] for k in combined_center_abnormal.keys()]))
        classifications = sorted(set([k[1] for k in combined_abnormal_class.keys()]))

        all_nodes = centers + abnormalities + classifications
        node_indices = {node: i for i, node in enumerate(all_nodes)}

        # 3. 创建连接数据 (优化显示效果)
        links = []

        # 医疗主题配色方案
        center_colors = ['#3498db', '#2ecc71', '#9b59b6']  # 蓝、绿、紫
        abnormal_colors = ['#e74c3c', '#f39c12', '#d35400', '#c0392b']  # 红色系
        class_colors = ['#16a085', '#27ae60', '#2980b9']  # 绿色/蓝色系

        # 中心→异常连接
        for (center, abnormal), count in combined_center_abnormal.items():
            if count >= min_count:
                links.append({
                    'source': node_indices[center],
                    'target': node_indices[abnormal],
                    'value': count,
                    'color': 'rgba(200, 200, 200, 0.4)',  # 半透明灰色连接线
                    'label': f"{center}→{abnormal}: {count}"
                })

        # 异常→分类连接
        for (abnormal, cls), count in combined_abnormal_class.items():
            if count >= min_count:
                links.append({
                    'source': node_indices[abnormal],
                    'target': node_indices[cls],
                    'value': count,
                    'color': 'rgba(200, 200, 200, 0.4)',
                    'label': f"{abnormal}→{cls}: {count}"
                })

        # 4. 创建Sankey图
        fig = go.Figure(go.Sankey(
            arrangement="snap",  # 优化节点布局
            node=dict(
                pad=30,  # 增加节点间距
                thickness=25,  # 更粗的节点
                line=dict(color="black", width=1),
                label=all_nodes,
                color=(
                        [center_colors[i % len(center_colors)] for i in range(len(centers))] +
                        [abnormal_colors[i % len(abnormal_colors)] for i in range(len(abnormalities))] +
                        [class_colors[i % len(class_colors)] for i in range(len(classifications))]
                ),
                hovertemplate='<b>%{label}</b><br>共 %{value} 个连接<extra></extra>',
                x=[0] * len(centers) + [0.5] * len(abnormalities) + [1] * len(classifications),  # 明确列位置
                y=[i / (len(centers) + 1) for i in range(1, len(centers) + 1)] +  # 均匀分布节点
                  [i / (len(abnormalities) + 1) for i in range(1, len(abnormalities) + 1)] +
                  [i / (len(classifications) + 1) for i in range(1, len(classifications) + 1)]
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color=[link['color'] for link in links],
                hoverlabel=dict(bgcolor="#34495e", font=dict(color="white")),
                hovertemplate='<b>%{label}</b><extra></extra>'
            )
        ))

        # 5. 添加专业样式
        fig.update_layout(
            title={
                'text': "<b>多中心医疗数据分析</b><br><sup>中心 → 异常发现 → 临床分类</sup>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color='#2c3e50', family="Arial")
            },
            font=dict(size=12, family="Arial"),
            width=1200,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=100, r=100, b=100, t=150),
            hovermode='x unified',
            annotations=[
                dict(
                    x=0, y=1.1,
                    xref='paper', yref='paper',
                    text='<b>医疗中心</b>',
                    showarrow=False,
                    font=dict(size=14, color=center_colors[0])
                ),
                dict(
                    x=0.5, y=1.1,
                    xref='paper', yref='paper',
                    text='<b>异常发现</b>',
                    showarrow=False,
                    font=dict(size=14, color=abnormal_colors[0])
                ),
                dict(
                    x=1, y=1.1,
                    xref='paper', yref='paper',
                    text='<b>临床分类</b>',
                    showarrow=False,
                    font=dict(size=14, color=class_colors[0])
                )
            ]
        )

        # 6. 保存或显示
        if save_path:
            fig.write_image(save_path, scale=2)  # 更高清的导出
            print(f"图表已保存至: {save_path}")
        else:
            fig.show()
class AllDatasets_cls_Seg_vstlge_numpy_QA_open(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_QA_open, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        #
        self.dataset_sax_QA = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False)
        self.dataset_fch_QA = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False)
        self.dataset_lge_QA = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False)

        # self.dataset_sax_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
        #                                prompt_mode='caption', use_seg=True, use_numpy = True)
        # self.dataset_fch_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
        #                                prompt_mode='caption', use_seg=True)
        # self.dataset_lge_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
        #                                prompt_mode='caption', use_seg=True)

        # self.dataset_sd = SD_Dataset_CMR_3D(self.args, self.tokenizer, self.mode)
        # self.dataset_loc = LocationDataset_CMR_3D(self.args, self.tokenizer, self.mode)
        # self.dataset_textsum = TextSumDataset_CMR(self.args, self.tokenizer, self.mode)
        # self.dataset_cls = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH','SAX', 'LGE'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls1 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls2 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX', 'LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='classification', use_seg=True, use_numpy=False)
        # self.dataset_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
        #                                prompt_mode='caption', use_seg=True)
        epoch = 1
        dataset_list = [
                           # self.dataset_3modal,
                           # self.dataset_sax,
                           # self.dataset_fch,
                           # self.dataset_lge,
                           # self.dataset_textsum,
                           # self.dataset_cls,
                           self.dataset_sax_QA,
                           self.dataset_fch_QA,
                           self.dataset_fch_QA,
                           self.dataset_fch_QA,
                           self.dataset_lge_QA,
                           self.dataset_lge_QA,
                           self.dataset_lge_QA,
                           self.dataset_lge_QA,
                           self.dataset_lge_QA,
                           # self.dataset_cls1,
                           # self.dataset_cls1,
                           # self.dataset_cls2,
                           # self.dataset_cls2,
                           # self.dataset_sax_caption,
                           # self.dataset_fch_caption,
                           # self.dataset_fch_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_caption,
                           # self.dataset_caption,
                       ] * epoch
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_QA_close(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_QA_close, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        #
        self.dataset_sax_QA = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_close', use_seg=True, use_numpy=False)
        self.dataset_fch_QA = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_close', use_seg=True, use_numpy=False)
        self.dataset_lge_QA = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_close', use_seg=True, use_numpy=False)

        # self.dataset_sax_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
        #                                prompt_mode='caption', use_seg=True, use_numpy = True)
        # self.dataset_fch_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
        #                                prompt_mode='caption', use_seg=True)
        # self.dataset_lge_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
        #                                prompt_mode='caption', use_seg=True)

        # self.dataset_sd = SD_Dataset_CMR_3D(self.args, self.tokenizer, self.mode)
        # self.dataset_loc = LocationDataset_CMR_3D(self.args, self.tokenizer, self.mode)
        # self.dataset_textsum = TextSumDataset_CMR(self.args, self.tokenizer, self.mode)
        # self.dataset_cls = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH','SAX', 'LGE'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls1 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls2 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX', 'LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='classification', use_seg=True, use_numpy=False)
        # self.dataset_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
        #                                prompt_mode='caption', use_seg=True)
        epoch = 1
        dataset_list = [
                           # self.dataset_3modal,
                           # self.dataset_sax,
                           # self.dataset_fch,
                           # self.dataset_lge,
                           # self.dataset_textsum,
                           # self.dataset_cls,
                           self.dataset_sax_QA,
                           self.dataset_fch_QA,
                           self.dataset_fch_QA,
                           self.dataset_fch_QA,
                           self.dataset_lge_QA,
                           self.dataset_lge_QA,
                           self.dataset_lge_QA,
                           self.dataset_lge_QA,
                           self.dataset_lge_QA,
                           # self.dataset_cls1,
                           # self.dataset_cls1,
                           # self.dataset_cls2,
                           # self.dataset_cls2,
                           # self.dataset_sax_caption,
                           # self.dataset_fch_caption,
                           # self.dataset_fch_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_lge_caption,
                           # self.dataset_caption,
                           # self.dataset_caption,
                       ] * epoch
        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_QA(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_QA, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        #
        self.dataset_sax_QA_close = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_close', use_seg=True, use_numpy=False, Multi_center='CD')
        self.dataset_fch_QA_close = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_close', use_seg=True, use_numpy=False, Multi_center='CD')
        self.dataset_lge_QA_close = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_close', use_seg=True, use_numpy=False, Multi_center='CD')
        self.dataset_All_QA_close = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH','SAX','LGE'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_close', use_seg=True, use_numpy=False, Multi_center='CD')

        self.dataset_sax_QA_open = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False, Multi_center='CD')
        self.dataset_fch_QA_open = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False, Multi_center='CD')
        self.dataset_lge_QA_open = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False, Multi_center='CD')
        self.dataset_All_QA_open = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH','SAX','LGE'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False, Multi_center='CD')

        # self.dataset_sax_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
        #                                prompt_mode='caption', use_seg=True, use_numpy = True)
        # self.dataset_fch_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
        #                                prompt_mode='caption', use_seg=True)
        # self.dataset_lge_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
        #                                prompt_mode='caption', use_seg=True)

        # self.dataset_sd = SD_Dataset_CMR_3D(self.args, self.tokenizer, self.mode)
        # self.dataset_loc = LocationDataset_CMR_3D(self.args, self.tokenizer, self.mode)
        # self.dataset_textsum = TextSumDataset_CMR(self.args, self.tokenizer, self.mode)
        # self.dataset_cls = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH','SAX', 'LGE'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls1 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls2 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX', 'LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='classification', use_seg=True, use_numpy=False)
        # self.dataset_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
        #                                prompt_mode='caption', use_seg=True)
        epoch = 1
        dataset_list = [
                       # self.dataset_All_QA_close,
                      self.dataset_All_QA_open
                       ] * 2 + [
                       self.dataset_sax_QA_open,
                      # self.dataset_sax_QA_close
                       ]+ [
                       # self.dataset_lge_QA_close,
                      self.dataset_lge_QA_open
                       ]*5+ [
                       self.dataset_fch_QA_open,
                      # self.dataset_fch_QA_close
                       ]*2

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_report(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_report, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        #

        self.dataset_All_report_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
                                                      dataset_names=['FCH', 'SAX', 'LGE'],
                                                      n_class=self.args.num_labels_D, prompt_mode='report',
                                                      use_seg=True, use_numpy=False, Multi_center='KM')
        self.dataset_All_report_cd = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
                                                         dataset_names=['FCH', 'SAX', 'LGE'],
                                                         n_class=self.args.num_labels_D, prompt_mode='report',
                                                         use_seg=True, use_numpy=False, Multi_center='CD')
        self.dataset_All_report_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
                                                        dataset_names=['FCH', 'SAX', 'LGE'],
                                                        n_class=self.args.num_labels_D, prompt_mode='report',
                                                        use_seg=True, use_numpy=False, Multi_center='SCS')
        self.dataset_All_QA_open_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH','SAX','LGE'],
                                       n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False, Multi_center='KM')
        self.dataset_All_QA_open_cd = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
                                                         dataset_names=['FCH', 'SAX', 'LGE'],
                                                         n_class=self.args.num_labels_D, prompt_mode='Question_open',
                                                         use_seg=True, use_numpy=False, Multi_center='CD')
        # self.dataset_All_QA_open_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                  dataset_names=['FCH', 'SAX', 'LGE'],
        #                                                  n_class=self.args.num_labels_D, prompt_mode='Question_open',
        #                                                  use_seg=True, use_numpy=False, Multi_center='SCS')
        self.dataset_All_QA_close_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
                                                         dataset_names=['FCH', 'SAX', 'LGE'],
                                                         n_class=self.args.num_labels_D, prompt_mode='Question_close',
                                                         use_seg=True, use_numpy=False, Multi_center='KM')
        self.dataset_All_QA_close_cd = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
                                                         dataset_names=['FCH', 'SAX', 'LGE'],
                                                         n_class=self.args.num_labels_D, prompt_mode='Question_close',
                                                         use_seg=True, use_numpy=False, Multi_center='CD')
        # self.dataset_All_QA_close_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                   dataset_names=['FCH', 'SAX', 'LGE'],
        #                                                   n_class=self.args.num_labels_D, prompt_mode='Question_close',
        #                                                   use_seg=True, use_numpy=False, Multi_center='SCS')
        self.dataset_All_caption_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
                                                           dataset_names=['FCH', 'SAX', 'LGE'],
                                                           n_class=self.args.num_labels_D, prompt_mode='caption',
                                                           use_seg=True, use_numpy=False, Multi_center='KM')
        self.dataset_sax_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
                                                      n_class=self.args.num_labels_D, prompt_mode='caption',
                                                      use_seg=True, use_numpy=False)
        self.dataset_fch_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
                                                      n_class=self.args.num_labels_D, prompt_mode='caption',
                                                      use_seg=True, use_numpy=False)
        self.dataset_lge_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
                                                      n_class=self.args.num_labels_D, prompt_mode='caption',
                                                      use_seg=True, use_numpy=False)
        # self.dataset_All_caption_close_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                    dataset_names=['FCH', 'SAX', 'LGE'],
        #                                                    n_class=self.args.num_labels_D, prompt_mode='caption',
        #                                                    use_seg=True, use_numpy=False, Multi_center='SCS')
        # self.dataset_All_caption_close_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                    dataset_names=['FCH', 'SAX', 'LGE'],
        #                                                    n_class=self.args.num_labels_D, prompt_mode='caption',
        #                                                    use_seg=True, use_numpy=False, Multi_center='SCS')


        epoch = 1
        dataset_list = [self.dataset_All_report_km,
                       self.dataset_All_report_cd,
                       self.dataset_All_report_scs,
                        self.dataset_All_QA_open_km,
                        self.dataset_All_QA_open_cd,
                        self.dataset_All_QA_close_km,
                        self.dataset_All_QA_close_cd,
                        self.dataset_All_caption_km,
                        self.dataset_sax_caption,
                        self.dataset_fch_caption,
                        self.dataset_lge_caption,
                        ]

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_report_LGE(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_report_LGE, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        #

        self.dataset_All_report_km = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                                      dataset_names=['FCH', 'SAX', 'LGE'],
                                                      n_class=self.args.num_labels_D, prompt_mode='report',
                                                      use_seg=True, use_numpy=False, Multi_center='KM', use_det =True)
        self.dataset_All_report_cd = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                                         dataset_names=['FCH', 'SAX', 'LGE'],
                                                         n_class=self.args.num_labels_D, prompt_mode='report',
                                                         use_seg=True, use_numpy=False, Multi_center='CD', use_det =True)
        self.dataset_All_report_scs = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                                        dataset_names=['FCH', 'SAX', 'LGE'],
                                                        n_class=self.args.num_labels_D, prompt_mode='report',
                                                        use_seg=True, use_numpy=False, Multi_center='SCS', use_det =True)
        # self.dataset_All_QA_open_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='Question_open', use_seg=True, use_numpy=False, Multi_center='KM')
        # self.dataset_All_QA_open_cd = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                  dataset_names=['LGE'],
        #                                                  n_class=self.args.num_labels_D, prompt_mode='Question_open',
        #                                                  use_seg=True, use_numpy=False, Multi_center='CD')
        # # self.dataset_All_QA_open_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        # #                                                  dataset_names=['FCH', 'SAX', 'LGE'],
        # #                                                  n_class=self.args.num_labels_D, prompt_mode='Question_open',
        # #                                                  use_seg=True, use_numpy=False, Multi_center='SCS')
        # self.dataset_All_QA_close_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                  dataset_names=['LGE'],
        #                                                  n_class=self.args.num_labels_D, prompt_mode='Question_close',
        #                                                  use_seg=True, use_numpy=False, Multi_center='KM')
        # self.dataset_All_QA_close_cd = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                  dataset_names=['LGE'],
        #                                                  n_class=self.args.num_labels_D, prompt_mode='Question_close',
        #                                                  use_seg=True, use_numpy=False, Multi_center='CD')
        # # self.dataset_All_QA_close_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        # #                                                   dataset_names=['FCH', 'SAX', 'LGE'],
        # #                                                   n_class=self.args.num_labels_D, prompt_mode='Question_close',
        # #                                                   use_seg=True, use_numpy=False, Multi_center='SCS')
        # self.dataset_All_caption_km = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                    dataset_names=['LGE'],
        #                                                    n_class=self.args.num_labels_D, prompt_mode='caption',
        #                                                    use_seg=True, use_numpy=False, Multi_center='KM')
        # self.dataset_sax_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX'],
        #                                               n_class=self.args.num_labels_D, prompt_mode='caption',
        #                                               use_seg=True, use_numpy=False)
        # self.dataset_fch_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH'],
        #                                               n_class=self.args.num_labels_D, prompt_mode='caption',
        #                                               use_seg=True, use_numpy=False)
        # self.dataset_lge_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['LGE'],
        #                                               n_class=self.args.num_labels_D, prompt_mode='caption',
        #                                               use_seg=True, use_numpy=False)
        # self.dataset_All_caption_close_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                    dataset_names=['FCH', 'SAX', 'LGE'],
        #                                                    n_class=self.args.num_labels_D, prompt_mode='caption',
        #                                                    use_seg=True, use_numpy=False, Multi_center='SCS')
        # self.dataset_All_caption_close_scs = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode,
        #                                                    dataset_names=['FCH', 'SAX', 'LGE'],
        #                                                    n_class=self.args.num_labels_D, prompt_mode='caption',
        #                                                    use_seg=True, use_numpy=False, Multi_center='SCS')


        epoch = 1
        dataset_list = [self.dataset_All_report_km,
                       self.dataset_All_report_cd,
                       self.dataset_All_report_scs,
                        # self.dataset_All_QA_open_km,
                        # self.dataset_All_QA_open_cd,
                        # self.dataset_All_QA_close_km,
                        # self.dataset_All_QA_close_cd,
                        # self.dataset_All_caption_km,
                        # self.dataset_sax_caption,
                        # self.dataset_fch_caption,
                        # self.dataset_lge_caption,
                        ]

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_Mix(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_Mix, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        dataset_cls_list = []
        dataset_report_list = []
        dataset_QA_list = []
        dataset_abnormal_list = []
        dataset_caption_list = []
        for center in ['CD', 'KM', 'SCS']:
            dataset_cls_list.append(
                UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                        dataset_names=['FCH','SAX', 'LGE'],
                                        n_class=self.args.num_labels_D, prompt_mode='classification',
                                        use_seg=True, use_numpy=False, Multi_center=center, use_det=True)

            )
            dataset_report_list.append(
                UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                   dataset_names=['FCH', 'SAX', 'LGE'],
                                   n_class=self.args.num_labels_D, prompt_mode='report',
                                   use_seg=True, use_numpy=False, Multi_center=center, use_det=True)
            )
            dataset_abnormal_list.append(
            UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                    dataset_names=['FCH', 'SAX', 'LGE'],
                                    n_class=self.args.num_labels_D, prompt_mode='abnormal_close',
                                    use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                                    )
            )
        for center in ['CD', 'KM']:
            for dataset_i in [['FCH', 'SAX', 'LGE'], ['FCH'], ['SAX'], ['LGE']]:
                dataset_QA_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                            dataset_names=dataset_i,
                                            n_class=self.args.num_labels_D, prompt_mode='Question_close',
                                            use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                                            )
                )
                dataset_QA_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                            dataset_names=dataset_i,
                                            n_class=self.args.num_labels_D, prompt_mode='Question_open',
                                            use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                                            )
                )
        for center in ['KM']:
            for dataset_i in [['FCH', 'SAX', 'LGE'], ['FCH'], ['SAX'], ['LGE']]:
                dataset_caption_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                       dataset_names=dataset_i,
                                       n_class=self.args.num_labels_D, prompt_mode='caption',
                                       use_seg=True, use_numpy=False, Multi_center=center, use_det=True))


        self.dataset_sd = SD_Dataset_CMR_3D(self.args, self.tokenizer, self.mode)
        self.dataset_loc = LocationDataset_CMR_3D(self.args, self.tokenizer, self.mode)
        self.dataset_textsum = TextSumDataset_CMR(self.args, self.tokenizer, self.mode)
        # self.dataset_cls = UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode, dataset_names=['FCH','SAX', 'LGE'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls1 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX'],
        #                                n_class = self.args.num_labels_D , prompt_mode='classification', use_seg=True, use_numpy = False)
        # self.dataset_cls2 = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['SAX', 'LGE'],
        #                                n_class=self.args.num_labels_D, prompt_mode='classification', use_seg=True, use_numpy=False)
        # self.dataset_caption = UniDatasets_tsvlge(self.args, self.tokenizer, self.mode, dataset_names=['FCH', 'SAX', 'LGE'],
        #                                prompt_mode='caption', use_seg=True)
        epoch = 1
        dataset_list = (dataset_cls_list*5+
                        dataset_report_list+
                        dataset_caption_list+
                        dataset_QA_list+
                        dataset_abnormal_list
                        )

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_Mix_MACE(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_Mix_MACE, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        dataset_cls_list = []

        for center in ['KM']:
            dataset_cls_list.append(
                UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                        dataset_names=['FCH','SAX', 'LGE'],
                                        n_class=self.args.num_labels_D, prompt_mode='mace',
                                        use_seg=True, use_numpy=False, Multi_center=center, use_det=True)

            )



        epoch = 1
        dataset_list = dataset_cls_list

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_VQA(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_VQA, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        dataset_cls_list = []
        dataset_report_list = []
        dataset_QA_list = []
        dataset_abnormal_list = []
        dataset_caption_list = []
        for center in ['CD', 'KM', 'SCS']:
            dataset_report_list.append(
                UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                   dataset_names=['FCH', 'SAX', 'LGE'],
                                   n_class=self.args.num_labels_D, prompt_mode='report',
                                   use_seg=True, use_numpy=False, Multi_center=center, use_det=True)
            )
            dataset_abnormal_list.append(
            UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                    dataset_names=['FCH', 'SAX', 'LGE'],
                                    n_class=self.args.num_labels_D, prompt_mode='abnormal_close',
                                    use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                                    )
            )
        for center in ['CD', 'KM']:
            for dataset_i in [['FCH', 'SAX', 'LGE'], ['FCH'], ['SAX'], ['LGE']]:
                dataset_QA_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                            dataset_names=dataset_i,
                                            n_class=self.args.num_labels_D, prompt_mode='Question_close',
                                            use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                                            )
                )
                dataset_QA_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                            dataset_names=dataset_i,
                                            n_class=self.args.num_labels_D, prompt_mode='Question_open',
                                            use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                                            )
                )
        for center in ['KM']:
            for dataset_i in [['FCH', 'SAX', 'LGE'], ['FCH'], ['SAX'], ['LGE']]:
                dataset_caption_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                       dataset_names=dataset_i,
                                       n_class=self.args.num_labels_D, prompt_mode='caption',
                                       use_seg=True, use_numpy=False, Multi_center=center, use_det=True))

        epoch = 1
        dataset_list = (dataset_report_list+
                        dataset_caption_list+
                        dataset_QA_list+
                        dataset_abnormal_list
                        )

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_test(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_test, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        dataset_cls_list = []
        dataset_report_list = []
        dataset_QA_list = []
        dataset_abnormal_list = []
        dataset_caption_list = []

        for center in ['CD', 'KM', 'SCS']:
            dataset_report_list.append(
                UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                        dataset_names=['FCH', 'SAX', 'LGE'],
                                        n_class=self.args.num_labels_D, prompt_mode='report',
                                        use_seg=True, use_numpy=False, Multi_center=center, use_det=True)
            )

        epoch = 1
        dataset_list = (
                        dataset_report_list
                        )

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_VQA_test(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_VQA_test, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        dataset_cls_list = []
        dataset_report_list = []
        dataset_QA_list = []
        dataset_abnormal_list = []
        dataset_caption_list = []

        for center in ['KM']:
            for dataset_i in [['FCH', 'SAX', 'LGE'], ['FCH'], ['SAX'], ['LGE']]:
                dataset_QA_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                            dataset_names=dataset_i,
                                            n_class=self.args.num_labels_D, prompt_mode='Question_open',
                                            use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                                            )
                )
                # dataset_QA_list.append(
                #     UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                #                             dataset_names=dataset_i,
                #                             n_class=self.args.num_labels_D, prompt_mode='Question_open',
                #                             use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                #                             )
                # )

        epoch = 1
        dataset_list = (
                        dataset_QA_list
                        )

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_VQAC_test(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_VQAC_test, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        dataset_cls_list = []
        dataset_report_list = []
        dataset_QA_list = []
        dataset_abnormal_list = []
        dataset_caption_list = []

        for center in ['CD','KM']:
            for dataset_i in [['FCH', 'SAX', 'LGE'], ['FCH'], ['SAX'], ['LGE']]:
                dataset_QA_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                            dataset_names=dataset_i,
                                            n_class=self.args.num_labels_D, prompt_mode='Question_close',
                                            use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                                            )
                )
                # dataset_QA_list.append(
                #     UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                #                             dataset_names=dataset_i,
                #                             n_class=self.args.num_labels_D, prompt_mode='Question_open',
                #                             use_seg=True, use_numpy=False, Multi_center=center, use_det=True,
                #                             )
                # )

        epoch = 1
        dataset_list = (
                        dataset_QA_list
                        )

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllDatasets_cls_Seg_vstlge_numpy_Cap(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(AllDatasets_cls_Seg_vstlge_numpy_Cap, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        dataset_cls_list = []
        dataset_report_list = []
        dataset_QA_list = []
        dataset_abnormal_list = []
        dataset_caption_list = []

        for center in ['KM']:
            for dataset_i in [['FCH', 'SAX', 'LGE'], ['FCH'], ['SAX'], ['LGE']]:
                dataset_caption_list.append(
                    UniDatasets_tsvlge_ALL2(self.args, self.tokenizer, self.mode,
                                       dataset_names=dataset_i,
                                       n_class=self.args.num_labels_D, prompt_mode='caption',
                                       use_seg=True, use_numpy=False, Multi_center=center, use_det=True))

        epoch = 1
        dataset_list = (
                        dataset_caption_list
                        )

        self.dataset = ConcatDataset(dataset_list)

        self.max_position_embeddings = 4096

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
