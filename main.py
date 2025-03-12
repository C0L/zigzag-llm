"""
Make a single plot for 1 model on 1 architecture
"""

import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from src.config import GPT3_175B, W8A8, W32A32
from src.export_onnx import Stage
from src.plots import plot_energy_and_latency
from src.simulation import run_simulation
from src.util import (
    CME_T,
    get_cmes_full_model_from_pickle,
    get_experiment_id,
)

model = GPT3_175B
model.batch_size = 1
model.prefill_size = 256
model.decode_size = 256
quant = W8A8
accelerator = "generic_array_8b"
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/main"


def run_experiment():
    for stage in Stage:
        run_simulation(
            model=model,
            stage=stage,
            quant=quant,
            accelerator_name=accelerator,
            mapping_path=mapping_path,
            output_dir=out_path,
        )


if __name__ == "__main__":
    run_experiment()

    cmes_per_group: list[list[CME_T]] = []

    for stage in Stage:
        experiment_name = get_experiment_id(model, stage, quant, accelerator)
        pickle_filename = f"{out_path}/{experiment_name}/cmes.pickle"
        cmes_full_model = get_cmes_full_model_from_pickle(pickle_filename, model, stage)
        cmes_per_group.append(cmes_full_model)

    latency_data, energy_data = plot_energy_and_latency(
        cmes_per_group,
        supergroups=["Prefill", "Decode"],
        title=f"{model.name} ({quant.name})",
        filename=f"{out_path}/energy_and_latency_{model.name}.png",
    )

    # Latency and energy from only decode stage
    latency_data = latency_data[1::]
    energy_data  = energy_data[3::]
    
    print("Cycle Count:", np.sum(latency_data[:,:,0]))

    mac_energy  = np.sum(energy_data[:,:,0]) * 1E-12   # pJ -> Joules
    mac_latency = np.sum(latency_data[:,:,0]) / 1.05E9 # cycles -> Seconds

    print("MAC ENERGY:", mac_energy)
    print("MAC LATENCY:", mac_latency)

    OPS   = 2*175E9 / (mac_latency) # OPS
    TOPS  = OPS / 1E12
    print("TOPS", TOPS)
    WATTS = mac_energy / mac_latency # J/s
    TOPSW = TOPS / WATTS
    print("TOPSW", TOPSW)


