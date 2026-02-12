# -*- coding: utf-8 -*-
import os
import pandas as pd
import re


def calculate_cardiac_weights(excel_path):
    if not excel_path or not os.path.exists(excel_path):
        return pd.DataFrame()
    """
    计算心脏MRI参数的异常权重

    参数:
        excel_path (str): Excel文件路径

    返回:
        DataFrame: 包含每个参数的权重信息
    """
    # 定义参数映射和异常列表
    abnormal_list = [
        'LVEDD', '左心房前后径', 'RVEDD', '右心房内径', '左心室壁厚度',
        '最厚处厚度', '变薄区域', 'LV室壁运动', '运动异常范围', '异常部位',
        '室壁瘤', 'LVEF', 'LVEDV', 'LVESV', 'RVEF', 'RVEDV', 'RVESV',
        '二尖瓣返流', '三尖瓣返流', '主动脉瓣返流', '左心室LGE', 'LGE部位',
        'LGE分布', 'LGE透壁程度', '右心室LGE', 'RV-LGE部位', '心包强化',
        '心肌脂肪浸润', '脂肪浸润部位', '血栓', '血栓部位', '血栓大小',
        '心包积液', '胸腔积液'
    ]

    name_mapping = {
        'LVEDD': 'LV End-Diastolic Diameter (LVEDD)',
        '左心房前后径': 'Left Atrial Anteroposterior Diameter',
        'RVEDD': 'RV End-Diastolic Diameter (RVEDD)',
        '右心房内径': 'Right Atrial Diameter',
        '左心室壁厚度': 'LV Wall Thickness',
        '最厚处厚度': 'Maximal Wall Thickness',
        '变薄区域': 'Regional Wall Thinning',
        'LV室壁运动': 'LV Wall Motion',
        '运动异常范围': 'Wall Motion Abnormality Extent',
        '异常部位': 'Location of Abnormalities',
        '室壁瘤': 'Ventricular Aneurysm',
        'LVEF': 'Left Ventricular Ejection Fraction (LVEF)',
        'LVEDV': 'LV End-Diastolic Volume (LVEDV)',
        'LVESV': 'LV End-Systolic Volume (LVESV)',
        'RVEF': 'Right Ventricular Ejection Fraction (RVEF)',
        'RVEDV': 'RV End-Diastolic Volume (RVEDV)',
        'RVESV': 'RV End-Systolic Volume (RVESV)',
        '二尖瓣返流': 'Mitral Regurgitation',
        '三尖瓣返流': 'Tricuspid Regurgitation',
        '主动脉瓣返流': 'Aortic Regurgitation',
        '左心室LGE': 'LV Late Gadolinium Enhancement (LGE)',
        'LGE部位': 'LGE Location',
        'LGE分布': 'LGE Distribution Pattern',
        'LGE透壁程度': 'LGE Transmural Extent',
        '右心室LGE': 'RV Late Gadolinium Enhancement (RV-LGE)',
        'RV-LGE部位': 'RV-LGE Location',
        '心包强化': 'Pericardial Enhancement',
        '心肌脂肪浸润': 'Myocardial Fatty Infiltration',
        '脂肪浸润部位': 'Fatty Infiltration Location',
        '血栓': 'Intracardiac Thrombus',
        '血栓部位': 'Thrombus Location',
        '血栓大小': 'Thrombus Size',
        '心包积液': 'Pericardial Effusion',
        '胸腔积液': 'Pleural Effusion'
    }

    # 读取Excel数据
    df = pd.read_excel(excel_path, engine='openpyxl')
    columns_as_lists = {column: df[column].tolist() for column in df.columns}

    # 初始化异常字典
    abnormal_dict = {name_mapping[key]: [] for key in name_mapping.keys()}

    # 处理每条记录
    for index, user_prompt in enumerate(columns_as_lists['Trans_4']):
        if type(columns_as_lists['Trans_4'][index]) == str:
            text = columns_as_lists['Trans_4'][index]
        else:
            continue

        for chinese_name, english_name in name_mapping.items():
            pattern = r'\s*"{}",\s*"abnormal":\s*(true|false)'.format(chinese_name)
            match = re.search(pattern, text, re.IGNORECASE)
            value = -1
            if match:
                abnormal_value = match.group(1).lower()
                value = 1 if abnormal_value == 'true' else 0
            abnormal_dict[english_name].append(value)

    # 计算异常率
    abnormal_df = pd.DataFrame(abnormal_dict)
    abnormal_stats = {}

    for col in abnormal_df.columns:
        valid_data = abnormal_df[col][abnormal_df[col] != -1]
        if len(valid_data) > 0:
            abnormal_rate = sum(valid_data == 1) / len(valid_data) * 100
            abnormal_stats[col] = abnormal_rate

    abnormal_series = pd.Series(abnormal_stats).sort_values(ascending=False)

    # 计算权重
    abnormal_rates = abnormal_series.values / 100
    normal_rates = 1 - abnormal_rates
    param_names = abnormal_series.index.tolist()

    class_weights = {}
    for i, param in enumerate(param_names):
        current_abnormal_rate = abnormal_rates[i]
        current_normal_rate = normal_rates[i]

        weight_abnormal = 1 / (current_abnormal_rate + 1e-5)
        weight_normal = 1 / (current_normal_rate + 1e-5)

        total = weight_abnormal + weight_normal
        class_weights[param] = {
            'abnormal_rate': current_abnormal_rate,
            'normal_rate': current_normal_rate,
            'weight_abnormal': weight_abnormal / total,
            'weight_normal': weight_normal / total,
            'sample_size': len(abnormal_df[param][abnormal_df[param] != -1])
        }

    weights_df = pd.DataFrame.from_dict(class_weights, orient='index')
    return weights_df


# 使用示例
if __name__ == "__main__":
    excel_path = os.environ.get("KM_EXCEL_PATH", "data/NEW2014.9-2024.12.15CMR_concept_add_add_step10_Question_oc_R1_T.xlsx")
    weights_df = calculate_cardiac_weights(excel_path)
    print(weights_df)
