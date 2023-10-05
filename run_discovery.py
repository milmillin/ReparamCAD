import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import time

from rpcad.discovery import (
    load_output_dir,
    run_greedy_and_resort,
    run_single_sort,
)

from init_models import initialize_model


def main():
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--algo", type=str)

    args = parser.parse_args()
    rundir = Path(args.dir)
    init_model: str = args.model
    algo: str = args.algo

    del args

    outdir = Path("./output_constraints") / rundir.name
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving to {outdir}")

    prefix = f"{algo}--{init_model}"
    start_time = time.time()

    model = initialize_model(init_model, device="cpu")
    data = load_output_dir(rundir, init_model, model)
    model_gpu = initialize_model(init_model, device="cuda")
    data_gpu = load_output_dir(rundir, init_model, model_gpu)

    if algo == "greedy-and-resort":
        res = run_greedy_and_resort(data, data_gpu)
    elif algo == "single":
        res = run_single_sort(data, data_gpu)
    else:
        raise ValueError(f"Invalid algo: {algo}")

    torch.save(res, outdir / f"{prefix}.pt")
    torch.save(data, outdir / f"{prefix}-data.pt")

    duration = time.time() - start_time
    torch.save(duration, outdir / f"{prefix}-time.pt")


if __name__ == "__main__":
    main()
