#!/usr/bin/env bash

bench=("nl1" "nl2" "jet" "steam" "exp" "watertank" "nl164" "nl264" "jet64" "steam64" "exp64" "watertank64" "vdp" "sine2d" "nonlin-osc")

for b in "${bench[@]}"
do
    python3 ../main.py -c $b/config.yaml
done
