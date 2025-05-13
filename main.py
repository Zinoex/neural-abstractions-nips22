# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from functools import partial
import multiprocessing
import os
import time
import warnings 

import torch
from torch import nn
import numpy as np

from benchmarks import read_benchmark
from cegis.cegis import Cegis, ScalarCegis
from cegis.verifier import DRealVerifier
from cegis.translator import Translator
from cli import get_config, parse_command
import polyhedrons
from utils import Timeout, save_net_dict, interpolate_error, get_partitions
from anal import Analyser
from neural_abstraction import NeuralAbstraction
from config import Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size, device=self.device))  # Create layer on device
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size, device=self.device))  # Create layer on device
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class VerificationWrapperNetwork(nn.Module):
    def __init__(self, net):
        super(VerificationWrapperNetwork, self).__init__()
        self.model = net.network

    def forward(self, x):
        return self.model(x)


def verify(res_queue, verify, network, translator):
    candidate = translator.translate(network)
    found, cex = verify(candidate)
    res_queue.put((found, cex))


def main(config: Config):
    benchmark = read_benchmark(config.benchmark)
    if config.target_error == 0:
        config.target_error = interpolate_error(
            benchmark.name, get_partitions(benchmark, config.widths, config.scalar))
    # c = Cegis(benchmark, config.target_error, config.widths,
    #             config)
    # T = config.timeout_duration if config.timeout else float("inf")

    verifier_type = DRealVerifier
    x = verifier_type.new_vars(benchmark.dimension)
    verifier = verifier_type(
        x,
        benchmark.dimension,
        benchmark.get_domain,
        verbose=config.verbose,
    )
    translator = Translator(x, verifier.relu)
    truef = np.array(benchmark.f(x)).reshape(-1, 1)

    network = SimpleNN(benchmark.dimension, config.widths, benchmark.dimension)
    path = os.path.join(BASE_DIR, "results/nets", config.model_path)
    network.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    network = VerificationWrapperNetwork(network)

    res_queue = multiprocessing.Queue()
    verifier_verify = partial(verifier.verify, truef, epsilon=[config.target_error for _ in range(benchmark.dimension)])

    timeout = 3600
    print("Benchmark: {}".format(benchmark.name))

    process = multiprocessing.Process(target=verify, args=(res_queue, verifier_verify, network, translator))

    t0 = time.perf_counter()
    process.start()

    # Wait for {timeout} seconds or until process finishes
    process.join(timeout)

    # If thread is still active
    if process.is_alive():
        print("running... let's kill it...")

        process.kill()
        process.join()
    else:
        print("finished... let's get the result...")
        t1 = time.perf_counter()
        delta_t = t1 - t0
        found, cex = res_queue.get()
        print("Result: {}, {}".format(found, cex))
        print("Verifier Timers: {} \n".format(delta_t))

    # print("Abstraction Timers: {} \n".format(NA.get_timer()))
    # print("The abstraction consists of {} modes".format(len(NA.locations)))
    # if "xml" in config.output_type:
    #     if config.initial:
    #         XI = polyhedrons.vertices2polyhedron(config.initial)
    #     else:
    #         XI = None
    #     if config.forbidden:
    #         XU = polyhedrons.vertices2polyhedron(config.forbidden) # Currently unused
    #     NA.to_xml(config.output_file, bounded_time=config.bounded_time, T=config.time_horizon, initial_state=XI)
    # if "csv" in config.output_type:
    #     a = Analyser(NA)
    #     a.record(config.output_file, config, res, delta_t)
    # if "plot" in config.output_type:
    #     if benchmark.dimension !=2:
    #         warnings.warn("Attempted to plot for n-dimensional system")
    #     else:  
    #         NA.plot(label=True)
    # if "pkl" in config.output_type:
    #     NA.to_pkl(config.output_file)



if __name__ == "__main__":
    c = get_config()
    torch.set_num_threads(1)
    for i in range(int(c.repeat)):
        c.seed += i
        torch.manual_seed(0 + c.seed)
        main(c)
