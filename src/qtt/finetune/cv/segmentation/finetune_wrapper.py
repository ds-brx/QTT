import os
import time

import pandas as pd
import yaml

from . import train

hp_list = [
    "model",
    "batch_size",
    "lr"
]

task_args = [
    "train-split",
    "val-split",
    "num-classes",
]

def finetune_script(
    job: dict,
    task_info: dict,
):  
    print("Running Segmentation FInetuning script")
    config = dict(job["config"])
    config_id = job["config_id"]
    fidelity = job["fidelity"]
    data_path = task_info["data-path"]
    output_path = task_info.get("output-path", ".")
    output_dir = os.path.join(output_path, str(config_id))
    
    config["data-path"] = data_path
    config["fidelity"] = fidelity
    
    parser = train.get_args_parser()

    print("Config Evaluated..")
    print(config)
    parser.set_defaults(**config)

    args = parser.parse_args(parser)

    start = time.time()
    try:
        result = train.main(args)
    except Exception as e:
        result = e
    end = time.time()
    try:
        summary = pd.read_csv(os.path.join(output_dir, "summary.csv"))
        eval_top1 = summary["eval_top1"].iloc[-1]
    except FileNotFoundError:
        result = "No summary.csv found"

    if result is not None:
        report = job.copy()
        report["score"] = 0
        report["cost"] = end - start
        report["status"] = False
        report["info"] = result
        return report

    report = job.copy()
    report["score"] = eval_top1 / 100
    report["cost"] = end - start
    report["status"] = True
    report["info"] = {"path": output_dir}

    return report
