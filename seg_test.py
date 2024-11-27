import pandas as pd
import numpy as np
import torch
from qtt import QuickOptimizer, QuickTuner
from qtt.predictors import PerfPredictor, CostPredictor
from qtt.objective.cv.segmentation import (
    extract_segmentation_task_info_metafeat, 
    finetune_script
)
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    OrConjunction,
    OrdinalHyperparameter,
)
from torchvision.datasets import VOCSegmentation 


def get_config_space():
    cs = ConfigurationSpace("cv-segmentation")
    model = Categorical(
        "model_name_or_path", 
        [
            "fcn_resnet50", 
            "lraspp_mobilenet_v3_large",
            "mobilenet",
            "beit",
            "segformer"
            ]
         )
    bs = OrdinalHyperparameter("per_device_train_batch_size", [1,2])
    lr = OrdinalHyperparameter("learning_rate", [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01])
    sched = Categorical("lr_scheduler_type", ["linear", "constant"])
    wd = OrdinalHyperparameter("weight_decay", [0, 1e-05, 0.0001, 0.001, 0.01, 0.1])
    w_ep = OrdinalHyperparameter("warmup_steps", [0, 5, 10])
    
    cs.add(model, bs, lr, sched, wd, w_ep)

    return cs


if __name__ == "__main__":
    
    cs = get_config_space()
    
    df = pd.read_csv("hpo_results.csv")

    cost = df["training_time"]
    curve = df.filter(regex=r'^epoch_(?:[1-9]|10)$')
    config = df.drop(columns=["training_time"] + curve.columns.tolist())

    X = config
    y = curve.values

    perf_predictor = PerfPredictor().fit(X,y)

    y = cost.values.reshape(-1, 1) 
    cost_predictor = CostPredictor().fit(X,y)
    
    print("Generate Optimiser")
    optimizer = QuickOptimizer(
        cs,
        max_fidelity=10,                # number of steps, e.g. epochs, must match the length of learning curves passed during fitting
        perf_predictor=perf_predictor,
        cost_predictor=cost_predictor,
        cost_aware=True,
        cost_factor=1.0,                # balances the importance of the cost-sensitivity by adjusting the scale of the predicted cost values, lower values
        acq_fn="ei",                    # acquisiton function to use
        explore_factor=0.1,             # xi value in the acquisition that balances exploration and exploitation
        patience=2,                     # early stopping for single configurations, if score does not improve by `tol`
        tol=0.001,                      # tolerance for early stopping
        refit=True,                     # whether the predictor is refitted during optimization
        refit_init_steps=32,            # how many inital steps to wait before we start with refitting
        refit_interval=1,               # interval of refit, 1 refits every step
        seed=42
    )
    
    print("Task Info and Meta Data")
    # fix
    task_info, metafeat = extract_segmentation_task_info_metafeat(
                            dataset_class=VOCSegmentation, 
                            root='/work/dlclarge2/dasb-Camvid/VOC_2007', 
                            year='2007', 
                        )
    print("Optimiser Setup")
    optimizer.setup(10, None)  # number of configurations to sample

    print("Tuner Set Up")
    tuner = QuickTuner(
        optimizer,
        finetune_script,  # script to finetune the configurations
    )

    print("Start Tuner")
    traj, runtime, history = tuner.run(task_info=task_info, fevals=100, time_budget=600)
    config_id, config, score, budget, cost, info = tuner.get_incumbent()
    print("Tuner Finished")

    print("*****   QUICKTUNE RESULTS   *****")
    print("=================================")
    print()
    print("Best configuration found:")
    print(f"Archtitecture: {config['model']}")
    print(f"Score: {score}")
    print(f"Number of epochs trained: {budget}")
    print(f"Cost per epoch: {cost}")
    print(f"Config: {' '.join([f'{k}={v}' for k, v in config.items()])}")
    print()
    print("---------------------------------")
    print()
    print(f"Total number of evaluated configs: {len(tuner.optimizer.evaled)}")
    print(f"Total number of evaluations: {len(traj)}")