#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=staging
#SBATCH --output=out.out


source activate comm

# python 1_plot_sl.py --log_folder1 ../condition3_generated/experiment2/condition_c/training_log --log_prefix1 log_lst \
# --log_folder2 ../condition3_generated/experiment2/condition_c/training_log_context --log_prefix2 log_lst \
# --log_folder3 ../condition3_generated/experiment2/condition_c/training_log_exp --log_prefix3 log_lst \
# --accuracy_type valid --output_plot acc_lst_sl_generated.png

# python 1_plot_rf.py --log_folder1 ../condition3_generated/experiment2/condition_c/training_log --log_prefix1 log_rf \
# --log_folder2 ../condition3_generated/experiment2/condition_c/training_log_context --log_prefix2 log_rf \
# --log_folder3 ../condition3_generated/experiment2/condition_c/training_log_exp --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_comm_rf_generated.png




seed_list=(111 222 333 444 555 666 777 888 999 123 234 345 456 567 678 789 891 912)

for seed in "${seed_list[@]}"; do
  input_file="../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/output_epoch30.txt"
  output_file="../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/epoch30.txt"

  if [ -f "$input_file" ]; then
    python 4_informativeness.py \
      --input_files "$input_file" \
      --output_file "$output_file" \
      --rl_or_sl >> ../condition3_generated/experiment2/condition_c/dump_context/informativeness.txt
  else
    echo "Skipping seed ${seed}: file not found"
  fi
done


for seed in "${seed_list[@]}"; do
  input_file="../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/output_epoch0.txt"
  output_file="../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/epoch0.txt"

  if [ -f "$input_file" ]; then
    python 4_informativeness.py \
      --input_files "$input_file" \
      --output_file "$output_file" \
      --rl_or_sl >> ../condition3_generated/experiment2/condition_c/dump_context/spk_informativeness.txt
  else
    echo "Skipping seed ${seed}: file not found"
  fi
done



################## informativeness for three training pipelines ###########
# seed_list=(111 222 333 444 555 666 777 888 999 123 234 345 456 567 678 789 891 912)

# for seed in 111 222 333 444 555 666 777 888 999
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/experiment2/condition_c/dump/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated/experiment2/condition_c/dump/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> ../condition3_generated/experiment2/condition_c/dump/informativeness.txt
# done

# for seed in "${seed_list[@]}"; 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> ../condition3_generated/experiment2/condition_c/dump_context/informativeness.txt
# done

# for seed in 111 222 333 444 555 666 777 888 999
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/experiment2/condition_c/dump_exp/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated/experiment2/condition_c/dump_exp/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> ../condition3_generated/experiment2/condition_c/dump_exp/informativeness.txt
# done




# # ################ prototypes ###############
# seeds=(111 123 222 345 444 555 567 666 678 777)  # a
# seeds=(111 123 333 345 444 555 567 666 777 912)  # b
seeds=(111 123 222 345 444 555 567 666 777 912)  # c
# epochs=$(seq 0 30)       # Epochs 0 through 30
epochs=(30)  # Only epoch 30

for seed in "${seeds[@]}"; do
    for epoch in $epochs; do
        python 2_prototype.py ../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/output_epoch${epoch}.txt \
            --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
            --csv_output prototypes_rf_human_epoch${epoch}.csv --rl

        python 2_prototype.py ../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/output_epoch${epoch}.txt \
            --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
            --csv_output prototypes_rf_human_epoch${epoch}.csv --rl

        python 2_prototype.py ../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed${seed}/output_epoch${epoch}.txt \
            --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
            --csv_output prototypes_rf_human_epoch${epoch}.csv --rl
    done
done
