import os
import time

import pandas as pd
import yaml

from . import train

def finetune_script(
    job: dict,
    task_info: dict,
):  
    print("Running Segmentation Finetuning script")
   
   # default arguments
    args = train.get_args_parser().parse_args()

    config = job["config"]
    index = job ["config_id"]
    fidelity = job["fidelity"]
    output_path = task_info.get("output-path", ".")
    output_dir = os.path.join(output_path, str(config_id))

    # static args update
    args.data_path = task_info["data-path"]
    args.dataset = task_info["dataset"]
    args.device = "cuda"
    args.batch_size = 8
    args.epochs = 50
    args.workers = 2
    args.output_dir = output_dir
    args.resume = os.path.join(output_dir, "last.pth.tar")

    # config update
    args.__dict__.update(config)

    start = time.time()
    try:
        result = train.main(args)
    except Exception as e:
        result = e
    end = time.time()

    report = job.copy()
    report["score"] = result["Score"]
    report["cost"] = result["Cost"]
    report["status"] = True
    report["info"] = {"path": output_dir}

    return report
