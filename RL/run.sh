#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ik620

# gammas=(0 0.2 0.4 0.6 0.8 1)

# # Loop over the list of oracle names and submit jobs
# for gamma in "${gammas[@]}"
# do
#     # Modify the config file
#     sed -i "s/gamma: .*/gamma: $gamma/" train.yml
#     python train.py
# done

paths=("./models/ray_tradingPG/flash_crash_order_size=100")
experiment_names=("experiment_flash_crash_order_size=1e2_with_passive")
for i in "${!paths[@]}"
do
    sed -i "s|path: .*|path: ${paths[$i]}|" test.yml
    sed -i "s|experiment_name: .*|experiment_name: ${experiment_names[$i]}|" test.yml
    python test.py
done
