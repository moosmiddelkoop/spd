
spd-run --experiments tms_5-2-id,tms_40-10-id,resid_mlp1,resid_mlp2,resid_mlp3 --sweep spd/scripts/gate_sweep_resweep_sparsity.yaml --project nathu-spd-sigmoid_sweeps --n_agents 16

spd-run --experiments tms_5-2-id,tms_40-10-id,resid_mlp1,resid_mlp2,resid_mlp3 --sweep spd/scripts/p_anneal_sweep.yaml --n_agents 12

spd-run --experiments tms_40-10-id,resid_mlp1,resid_mlp2,resid_mlp3 --sweep spd/scripts/p_anneal_pnorm1_baseline.yaml --n_agents 8

spd-run --experiments mem_32_2x --sweep spd/scripts/memorization_hparam_sweep.yaml --n_agents 6

spd-run --experiments mem_32_2p8x --sweep spd/scripts/memorization_hparam_sweep.yaml --n_agents 4