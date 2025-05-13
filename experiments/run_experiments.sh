#!/usr/bin/env bash

# bench=("nl1" "nl2" "jet" "steam" "nest" "watertank")
bench=("nl164" "nl264" "jet64" "steam64" "nest64" "watertank64") # "vdp" "sine2d" "nonlin-osc")
cd nl1
for b in "${bench[@]}"
do
    cd ../$b
    ./table-1-row.sh
done
