import os
import time
import pandas as pd
from hf_pipeline import run_semantic_segmentation
from torch_pipeline import train
from sam_pipeline import run_sam  
import random
import itertools
import argparse

sam = [
    "sam"
]

hf_models = [
    "nvidia/mit-b0",
     "google/deeplabv3_mobilenet_v2_1.0_513",
     "microsoft/beit-base-finetuned-ade-640-640",
     "Intel/dpt-large-ade",
     "facebook/data2vec-vision-base",
     "apple/deeplabv3-mobilevit-small",
     "apple/mobilevitv2-1.0-imagenet1k-256",
     "openmmlab/upernet-convnext-tiny"
]

pt_models = [
    "fcn_resnet50", 
    "fcn_resnet101", 
    "deeplabv3_resnet50", 
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large", 
    "lraspp_mobilenet_v3_large"
]

hp_list = [
    "model_name_or_path",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "learning_rate",
    "lr_scheduler_type",
    "weight_decay",
    "warmup_steps",
]

static_args = [
    "--dataloader_num_workers", "2",
    "--do_train", "True",
    "--do_eval", "True",
    "--eval_strategy", "epoch"
]

def finetune_script(job: dict, task_info: dict):
    """
    Function to fine-tune a model based on the job and task info provided.
    
    Args:
    - job (dict): Configuration and other metadata for the job.
    - task_info (dict): Information about the task, including dataset and output path.

    Returns:
    - dict: A report containing results of the fine-tuning process.
    """
    # Extract job configuration and task information
    config = job.get("config", {})
    config_id = job.get("config_id", None)
    fidelity = job.get("fidelity", 1)
    dataset_name = task_info.get("dataset_name", None)
    data_path = task_info.get("data_path", None)
    output_path = task_info.get("output_dir", ".")
    
    # Ensure the essential information is present
    if not config or not data_path:
        raise ValueError("Missing required fields in job or task_info.")
    
    output_dir = os.path.join(output_path, str(config_id) if config_id else "default_output")

    # Initialize the arguments list
    args = []

    # Add regular hyperparameters from the config
    for hp in hp_list:
        if hp in config:
            args.append(f"--{hp}")
            args.append(str(config[hp]))

    # Add additional arguments
    args.extend([
        "--num_train_epochs", str(fidelity),
        "--dataset_name", dataset_name,
        "--output_dir", output_dir
    ])

    # Add static arguments (common settings)
    args.extend(static_args)

    # Choose the correct model pipeline based on the configuration
    if config.get("model_name_or_path") in hf_models:
        print(f"Using Hugging Face model: {config['model_name_or_path']}")
        args.extend(["--remove_unused_columns" , False])
        train_metrics, eval_metrics = run_semantic_segmentation.main(args)
        score = eval_metrics["eval_mean_iou"]
        cost = train_metrics["train_runtime"]
    
    elif config.get("model_name_or_path") in pt_models:
        print(f"Using PyTorch model: {config['model_name_or_path']}")
        parser = train.get_args_parser()
        args, _ = parser.parse_known_args(args)
        score, cost = train.main(args)
    elif config.get("model_name_or_path") in sam:
        parser = run_sam.get_args(args)
        args, _ = parser.parse_known_args(args)
        score,cost = run_sam.main(args)

    else:
        print(f"Unknown model: {config.get('model_name_or_path')}")
        score = cost = 0

    # Prepare the report (you may want to process `results` to extract actual scores/cost)
    report = job.copy()
    report["score"] = score
    report["cost"] = cost
    report["status"] = True  # Could be determined based on results
    report["info"] = {"path": output_dir}

    return report


if __name__ == "__main__":
    # Hyperparameter grid
    hyperparameter_grid = {
        "model_name_or_path": [
            "openmmlab/upernet-convnext-tiny",
        ],
        "per_device_train_batch_size": [1],
        "per_device_eval_batch_size": [1],
        "learning_rate": [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        "lr_scheduler_type": ["linear", "constant"],
        "weight_decay": [0, 1e-05, 0.0001, 0.001, 0.01, 0.1],
        "warmup_steps": [0, 5, 10]
    }

    keys, values = zip(*hyperparameter_grid.items())
    all_combinations = list(itertools.product(*values))

    random_combinations = random.sample(all_combinations, 1)

    dict_combinations = [
        {key: value for key, value in zip(keys, combination)}
        for combination in random_combinations
    ]

    file_path = "finetuning_results.csv"
    results = []
    for i, combination_dict in enumerate(dict_combinations, 1):
        job = {
            "config" : combination_dict,
            "config_id" : i,
            "fidelity" : 1
        }

        task_info = {
            "dataset_name" : "segments/sidewalk-semantic",
            "data_path" : "/home/dasb/workspace/cache/huggingface",
            "output_dir" : "./outputs"
        }

        report = finetune_script(job, task_info)
        result = combination_dict.copy()
        result["score"] = report["score"]
        result["cost"] = report["cost"]
        results.append(result)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(file_path, index=False)
    print(df)



    
