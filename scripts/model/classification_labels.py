"""Lightweight class-label definitions required by the archived classifiers.

These constants used to be imported from the training dataset module.  That
module reads site-specific spreadsheets at import time, which makes model-only
inference impossible on another machine.  Keeping the unchanged class order
here preserves the checkpoint architecture without importing any clinical
data or training-time dependencies.
"""

CLASSES_CN_3 = [
    "扩张型心肌病",
    "肥厚型心肌病",
    "正常",
    "心肌梗死",
    "高血压心脏病",
    "心肌炎",
]

CLASSES_CN_4 = [
    "扩张型心肌病",
    "肥厚型心肌病",
    "心肌炎",
    "心肌梗死",
    "高血压心脏病",
    "正常",
    "心肌淀粉样变",
    "传导系统疾病",
    "心肌致密化不全",
    "应激性心肌病",
    "致心律失常性",
    "心包炎",
    "肺动脉高压",
    "心脏瓣膜病",
    "先天性心脏病",
    "肿瘤相关",
    "产褥期相关心肌病",
    "系统性疾病相关心脏病",
    "卒中",
]
