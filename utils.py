"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

#from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import math
import numpy as np
import os
from downstream_dataset import TUABLoader, TUEVLoader, TUSLLoader, HMCLoader, WorkloadLoader,MODMALoader,OpenslrLoader
from metrics import binary_metrics_fn, multiclass_metrics_fn
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from typing import List, Union, Optional


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def prepare_TUEV_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_MODMA_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = MODMALoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = MODMALoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = MODMALoader(os.path.join(root, "val"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_Openslr_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = OpenslrLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = OpenslrLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = OpenslrLoader(os.path.join(root, "val"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUSL_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUSLLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUSLLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUSLLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_HMC_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = HMCLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = HMCLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = HMCLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_Workload_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = WorkloadLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = WorkloadLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = WorkloadLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def get_metrics(output, target, metrics, is_binary):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results

def calculate_batch_bleu_nltk(
    generated_batch: List[List[str]], 
    target_batch: List[List[str]],
    weights: tuple = (0.25, 0.25, 0.25, 0.25),
    smoothing: bool = True,
    use_corpus_bleu: bool = False
) -> dict:
    """
    使用NLTK计算两个batch之间的BLEU分数
    
    Args:
        generated_batch: 模型生成的batch文本，每个元素是token列表
        target_batch: 目标batch文本，每个元素是token列表
        weights: n-gram权重，默认为均匀分布
        smoothing: 是否使用平滑处理
        use_corpus_bleu: 是否使用语料级BLEU计算
    
    Returns:
        包含各种BLEU分数的字典
    """
    # 确保两个batch长度相同
    assert len(generated_batch) == len(target_batch), "两个batch的长度必须相同"
    
    # 准备数据格式
    references = [[target] for target in target_batch]  # 每个句子可以有多个参考，这里只有一个
    hypotheses = generated_batch
    
    # 平滑函数
    smoothie = SmoothingFunction().method1 if smoothing else None
    
    # 计算语料级BLEU（整体）
    corpus_bleu_score = corpus_bleu(
        references, 
        hypotheses, 
        weights=weights,
        smoothing_function=smoothie
    )
    
    return corpus_bleu_score

