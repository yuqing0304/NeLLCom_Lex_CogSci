#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --output=out.out



source activate balance

test_color="all"
seed_list=(111 123 333 345 444 555 567 666 777 912) 
total_epochs=30


################# sec1: train agents ###############
python train.py \
    --data_path "../color_game/data/data_gray/${test_color}/condition_bupsample200.csv" \
    --n_epochs 30 \
    --total_epochs 30 \
    --n_receivers 1 \
    --if_context \
    --test_color "${test_color}" \
    --seeds "${seed_list[@]}"



# ################# sec2: informativeness ###############
# seed_list=(111 123 333 345 444 555 567 666 777 912)  
seed_list=(111)   
total_epochs=30
condition_dirs=(
  # all/condition_b_1lst
  # all/condition_b_5lst
  # all/condition_b_30lst
  # all/condition_bupsample100_1lst
  all/condition_bupsample200_1lst
  # all/condition_bupsample100_5lst
  # all/condition_bupsample200_5lst
  # all/condition_bupsample100_30lst
  # all/condition_bupsample200_30lst

  # all/condition_b_15lst
  # all/condition_bupsample100_15lst
  # all/condition_bupsample200_15lst
)

for cond_path in "${condition_dirs[@]}"; do
  cond_name=$(basename "$cond_path")   # e.g. condition_b_1lst

  # 从目录名中解析 lst 数字（如 1lst → 1）
  lst=${cond_name##*_}        # 1lst / 5lst
  n=${lst%lst}

  max_epoch=$(( total_epochs / n ))

  echo "Processing $cond_name (lst=$n)"

  for epoch_num in 0 $max_epoch; do
    for seed in "${seed_list[@]}"; do
      python ../color_gamezero/analysis_ARR/4_informativeness.py \
        --input_files \
        "./${cond_path}/dump_context/msg_rf_seed${seed}/output_epoch${epoch_num}.txt" \
        --output_file \
        "./${cond_path}/dump_context/msg_rf_seed${seed}/epoch${epoch_num}.txt" \
        --rl_or_sl >> \
        "./${cond_path}/dump_context/informativeness.txt"
    done
  done
done



# ################# sec3: prototypes ###############
# seed_list=(111 123 333 345 444 555 567 666 777 912)  
seed_list=(111)   
total_epochs=30

for cond_path in "${condition_dirs[@]}"; do
  cond_name=$(basename "$cond_path")

  lst=${cond_name##*_}
  n=${lst%lst}
  max_epoch=$(( total_epochs / n ))

  echo "Prototypes for $cond_name (lst=$n)"

  for seed in "${seed_list[@]}"; do
    for epoch in $(seq 0 $max_epoch); do
      python ../color_gamezero/analysis_ARR_latest/2_prototype_group.py \
        "./${cond_path}/dump_context/msg_rf_seed${seed}/output_epoch${epoch}.txt" \
        --pickle_output "prototypes_rf_human_epoch${epoch}.pkl" \
        --csv_output "prototypes_rf_human_epoch${epoch}.csv" \
        --rl
    done
  done
done


################# sec4: drift ###############
# seed_list=(111 123 333 345 444 555 567 666 777 912)  
seed_list=(111)   
total_epochs=30

for cond_path in "${condition_dirs[@]}"; do
  cond_name=$(basename "$cond_path")   # e.g. condition_b_1lst

  lst=${cond_name##*_}                 # 1lst / 5lst
  n=${lst%lst}
  last_epoch_n=$(( total_epochs / n ))

  python ../color_gamezero/analysis_ARR/2_avg_drift.py \
    --epoch $last_epoch_n \
    --seeds "${seed_list[@]}" \
    --label_prototype ../color_gamezero/analysis_ARR/prototypes.pkl \
    --base_dirs \
      dump_context=./${cond_path}/dump_context/msg_rf_seed{seed}/ \
    --last_epoch_num $last_epoch_n \
    --output ./all/drift_${cond_name}.csv
done


































# path_color="condition_bupsample100"
# path_color="condition_boliveupsample100"
# data_path_full =../color_game/data/data_gray/${path_color}.csv



# ############ test: group communication ##################
# python train.py --data_path ../color_game/data/data_barney/condition_bbarney70upsample100.csv --n_epochs 30 --total_epochs 30 --n_receivers 1 --if_context
# python train.py --data_path data_path_full --n_epochs 30 --total_epochs 30 --n_receivers 1 --if_context
# python train.py --data_path ../color_game/data/data_gray/${path_color}.csv --n_epochs 1 --total_epochs 1 --n_receivers 1 --if_context
# python train.py --data_path ../color_game/data/data_gray/condition_bupsample200.csv --n_epochs 30 --total_epochs 30 --n_receivers 1 --if_context


############ test: group communication ##################
# python train.py --data_path ./data/condition_b.csv --n_epochs 30 --total_epochs 30 --n_receivers 1 --if_context
# python train.py --data_path ./data/condition_b.csv --n_epochs 30 --total_epochs 30 --n_receivers 2 --if_context
# python train.py --data_path ./data/condition_b.csv --n_epochs 30 --total_epochs 30 --n_receivers 6 --if_context
# python train.py --data_path ./data/condition_b.csv --n_epochs 30 --total_epochs 30 --n_receivers 10 --if_context


############ experiment1 ##################
# python train.py --data_path ./data/different_base_data/condition_slrl2.csv --n_epochs 30 --n_comm_epochs 30 --if_context
# python train.py --data_path ./data/different_base_data/condition_slrl2.csv --n_epochs 1 --n_comm_epochs 30 


############ experiment2 ##################
# python train.py --data_path ./data/condition_a.csv --n_epochs 30 --n_comm_epochs 30 --if_context
# python train.py --data_path ./data/condition_a.csv --n_epochs 30 --n_comm_epochs 30 

# python train.py --data_path ./data/condition_b.csv --n_epochs 30 --n_comm_epochs 30 --if_context
# python train.py --data_path ./data/condition_b.csv --n_epochs 30 --n_comm_epochs 30 

# python train.py --data_path ./data/condition_c.csv --n_epochs 30 --n_comm_epochs 30 --if_context
# python train.py --data_path ./data/condition_c.csv --n_epochs 30 --n_comm_epochs 30 

############ qualitative eval experiment ##################
# python train.py --data_path ./data/condition_slrlhuman.csv --n_epochs 10 --n_comm_epochs 20 --if_context

# mv ./training_log ./training_log_context ./dump ./dump_context ../color_gamezero
