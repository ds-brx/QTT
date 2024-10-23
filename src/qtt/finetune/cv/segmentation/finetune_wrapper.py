import os
import time

import pandas as pd
import yaml

from . import train

hp_list = [
    "batch_size",
    "epochs",
    "lr",
    "momentum",
    "weight_decay",
    "lr_warmup_epochs",
    "lr_warmup_method",
    "lr_warmup_decay",
    "print_freq",
    "output_dir",
    "resume",
    "start_epoch",
    "dataset",
    "model",
    "workers"
]

num_hp_list = [
    "aux_loss",
    "test_only",
    "use_deterministic_algorithms",
    "amp",
    "use_v2"
]

bool_hp_list = [
    "aux_loss",
    "test_only",
    "use_deterministic_algorithms",
    "amp",
    "use_v2"
]

cond_hp_list =[
    "world_size",
    "dist_url"
]

static_args = [
    "--data-path",
    "--output-dir",
    "--resume",
    "--weights",
    "--weights-backbone",
    "--backend"
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
    config = dict(job["config"])
    config_id = job["config_id"]
    fidelity = job["fidelity"]
    data_path = task_info["data-path"]
    output_path = task_info.get("output-path", ".")

    args = [data_path]
    # REGULAR HPS/ARGS
    for hp in hp_list:
        if hp in config:
            args += [f"--{hp}", str(config[hp])]

    # NUMERICAL ARGS (if the value is not 0)
    for hp in num_hp_list:
        value = config.get(hp)
        args += [f"--{hp}", str(value)]

    # BOOLEAN ARGS
    for hp in bool_hp_list:
        enabled = config.get(hp, False)
        if enabled:
            args += [f"--{hp}"]

    # CONDITIONAL ARGS
    for hp in cond_hp_list:
        option = config.get(hp, False)
        if option:
            args += [f"--{hp}", str(option)]


    args += ["--fidelity", str(fidelity)]
    args += ["--experiment", str(config_id)]
    args += ["--output", output_path]

    # OUTPUT DIRECTORY
    output_dir = os.path.join(output_path, str(config_id))
    resume_path = os.path.join(output_dir, "last.pth.tar")
    if os.path.exists(resume_path):
        args += ["--resume", resume_path]

    args += static_args

    parser = train.build_parser()
    args, _ = parser.parse_known_args(args)
    args_text = yaml.safe_dump(args.__dict__)

    start = time.time()
    try:
        result = train.main(args, args_text)
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
