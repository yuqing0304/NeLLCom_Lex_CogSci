#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=staging
#SBATCH --output=out.out

# TODO: 
# variation, visualization: many sl +rl results for several seeds, several exploration param
# upsample rare colors, check whether they learn these words
# subsample frequent colors ~50?

# NOTE: for different seeds, the data split is the same)

#### #SBATCH --gpus=1 #SBATCH --partition=gpu_a100


source activate comm

# python 1_plot_sl.py --log_folder1 ../condition3_generated/experiment1/training_log --log_prefix1 log_lst \
# --log_folder2 ../condition3_generated/experiment1/training_log_context --log_prefix2 log_lst \
# --log_folder3 ../condition3_generated/experiment1/training_log_exp --log_prefix3 log_lst \
# --accuracy_type valid --output_plot acc_lst_sl_generated.png

# python 1_plot_rf.py --log_folder1 ../condition3_generated/experiment1/training_log --log_prefix1 log_rf \
# --log_folder2 ../condition3_generated/experiment1/training_log_context --log_prefix2 log_rf \
# --log_folder3 ../condition3_generated/experiment1/training_log_exp --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_comm_rf_generated.png




################## informativeness for three training pipelines ###########

# for seed in 111 222 333 444 555 666 777 888 999
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/experiment1/dump/msg_rf_seed${seed}/output_epoch10.txt \
#     --output_file ../condition3_generated/experiment1/dump/msg_rf_seed${seed}/epoch10.txt \
#     --rl_or_sl >> ../condition3_generated/experiment1/dump/informativeness.txt
# done

# for seed in 111 222 333 444 555 666 777 888 999
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/experiment1/dump_context/msg_rf_seed${seed}/output_epoch10.txt \
#     --output_file ../condition3_generated/experiment1/dump_context/msg_rf_seed${seed}/epoch10.txt \
#     --rl_or_sl >> ../condition3_generated/experiment1/dump_context/informativeness.txt
# done

# for seed in 111 222 333 444 555 666 777 888 999
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/experiment1/dump_exp/msg_rf_seed${seed}/output_epoch10.txt \
#     --output_file ../condition3_generated/experiment1/dump_exp/msg_rf_seed${seed}/epoch10.txt \
#     --rl_or_sl >> ../condition3_generated/experiment1/dump_exp/informativeness.txt
# done



# # ################ prototypes ###############
# seeds=(111 222 333 444 555 666 777 888 999)  # Add more seed numbers as needed
# epochs=$(seq 0 30)       # Epochs 0 through 30

# for seed in "${seeds[@]}"; do
#     for epoch in $epochs; do
#         python 2_prototype.py ../condition3_generated/test/dump_context/msg_rf_seed${seed}/output_epoch${epoch}.txt \
#             --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
#             --csv_output prototypes_rf_human_epoch${epoch}.csv --rl

#         python 2_prototype.py ../condition3_generated/test/dump/msg_rf_seed${seed}/output_epoch${epoch}.txt \
#             --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
#             --csv_output prototypes_rf_human_epoch${epoch}.csv --rl

#         python 2_prototype.py ../condition3_generated/test/dump_exp/msg_rf_seed${seed}/output_epoch${epoch}.txt \
#             --pickle_output prototypes_rf_human_epoch${epoch}.pkl \
#             --csv_output prototypes_rf_human_epoch${epoch}.csv --rl
#     done
# done






# python 2_drift.py
# python 2_prototype_table.py

# ### Accuracy and communicative success
# #### Figure with speaker accuracy on the test set through training on the human data. Two line graphs side by side: one for SL, the other for RL1.

# ##### speaker accuracy for SL
# python 1_plot_sl.py --log_folder1 ../condition1_human/training_log --log_prefix1 log_spk \
# --log_folder2 ../condition1_human/training_log_context --log_prefix2 log_spk \
# --log_folder3 ../condition1_human/training_log_exp --log_prefix3 log_spk \
# --accuracy_type valid --output_plot acc_spk_sl.png

# ##### listener accuracy for SL
# python 1_plot_sl.py --log_folder1 ../condition1_human/training_log --log_prefix1 log_lst \
# --log_folder2 ../condition1_human/training_log_context --log_prefix2 log_lst \
# --log_folder3 ../condition1_human/training_log_exp --log_prefix3 log_lst \
# --accuracy_type valid --output_plot acc_lst_sl.png


# ##### speaker accuracy for SL
# python 1_plot_sl.py --log_folder1 ../condition3_generated/training_log --log_prefix1 log_spk \
# --log_folder2 ../condition3_generated/training_log_context --log_prefix2 log_spk \
# --log_folder3 ../condition3_generated/training_log_exp --log_prefix3 log_spk \
# --accuracy_type valid --output_plot acc_spk_sl_generated.png

# ##### listener accuracy for SL
# python 1_plot_sl.py --log_folder1 ../condition3_generated/training_log --log_prefix1 log_lst \
# --log_folder2 ../condition3_generated/training_log_context --log_prefix2 log_lst \
# --log_folder3 ../condition3_generated/training_log_exp --log_prefix3 log_lst \
# --accuracy_type valid --output_plot acc_lst_sl_generated.png


# # ##### speaker accuracy for RL1
# # python 1_plot_rf_spk.py

# # #### Figure with communicative success across epochs in RL1. Two line graphs side by side: one with the human dataset, one with the generated dataset.

# # ##### the human dataset
# # python 1_plot_rf.py --log_folder1 ../condition1_human/training_log --log_prefix1 log_rf \
# # --log_folder2 ../condition1_human/training_log_context --log_prefix2 log_rf \
# # --log_folder3 ../condition1_human/training_log_exp --log_prefix3 log_rf \
# # --accuracy_type valid --output_plot acc_comm_rf_human.png

# ##### the generated dataset
# python 1_plot_rf.py --log_folder1 ../condition3_generated/training_log --log_prefix1 log_rf \
# --log_folder2 ../condition3_generated/training_log_context --log_prefix2 log_rf \
# --log_folder3 ../condition3_generated/training_log_exp --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_comm_rf_generated.png


# ##### ##### ##### ##### ##### ##### #####  for test purpose ##### ##### ##### ##### ##### ##### 
# python 1_plot_sl.py --log_folder1 ../condition3_generated/condition_slrl3/training_log_context --log_prefix1 log_spk \
# --log_folder2 ../condition3_generated/condition_slrl3/training_log_context --log_prefix2 log_spk \
# --log_folder3 ../condition3_generated/condition_slrl3/training_log_context --log_prefix3 log_spk \
# --accuracy_type valid --output_plot acc_spk_sl_generated.png

# python 1_plot_sl.py --log_folder1 ../condition3_generated/condition_slrl3/training_log_context --log_prefix1 log_lst \
# --log_folder2 ../condition3_generated/condition_slrl3/training_log_context --log_prefix2 log_lst \
# --log_folder3 ../condition3_generated/condition_slrl3/training_log_context --log_prefix3 log_lst \
# --accuracy_type valid --output_plot acc_lst_sl_generated.png

# python 1_plot_rf.py --log_folder1 ../condition3_generated/condition_slrl3/training_log_context --log_prefix1 log_rf \
# --log_folder2 ../condition3_generated/condition_slrl3/training_log_context --log_prefix2 log_rf \
# --log_folder3 ../condition3_generated/condition_slrl3/training_log_context --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_comm_rf_generated.png



# python 1_plot_sl.py --log_folder1 ../condition_allhuman/training_log_context --log_prefix1 log_spk \
# --log_folder2 ../condition_allhuman/training_log_context --log_prefix2 log_spk \
# --log_folder3 ../condition_allhuman/training_log_context --log_prefix3 log_spk \
# --accuracy_type valid --output_plot acc_spk_sl_human.png

# python 1_plot_sl.py --log_folder1 ../condition_allhuman/training_log_context --log_prefix1 log_lst \
# --log_folder2 ../condition_allhuman/training_log_context --log_prefix2 log_lst \
# --log_folder3 ../condition_allhuman/training_log_context --log_prefix3 log_lst \
# --accuracy_type valid --output_plot acc_lst_sl_human.png

# python 1_plot_rf.py --log_folder1 ../condition_allhuman/training_log_context --log_prefix1 log_rf \
# --log_folder2 ../condition_allhuman/training_log_context --log_prefix2 log_rf \
# --log_folder3 ../condition_allhuman/training_log_context --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_comm_rf_human.png

# python 4_informativeness.py \
#   --input_files ../condition3_generated/dump_context/msg_spk_seed111/sender_epoch29.txt \
#   --output_file ../condition3_generated/dump_context/msg_spk_seed111/epochspk30_agent.txt \
#   > condition3_generated_agent_sl_informativeness.txt

# python 4_informativeness.py \
#   --input_files ../condition3_generated/dump_context/msg_rf_seed111/output_epoch30.txt \
#   --output_file ../condition3_generated/dump_context/msg_rf_seed111/epoch30.txt \
#     --rl_or_sl > condition3_generated_agent_rf_informativeness.txt
# ##### ##### ##### ##### ##### ##### #####  for test purpose ##### ##### ##### ##### ##### ##### 

# ##### ##### ##### ##### ##### ##### #####  for test purpose ##### ##### ##### ##### ##### ##### 
# python 1_plot_sl.py --log_folder1 ../condition1_human/training_log_context --log_prefix1 log_spk \
# --log_folder2 ../condition1_human/training_log_context --log_prefix2 log_spk \
# --log_folder3 ../condition1_human/training_log_context --log_prefix3 log_spk \
# --accuracy_type valid --output_plot acc_spk_sl_human.png

# python 1_plot_sl.py --log_folder1 ../condition1_human/training_log_context --log_prefix1 log_lst \
# --log_folder2 ../condition1_human/training_log_context --log_prefix2 log_lst \
# --log_folder3 ../condition1_human/training_log_context --log_prefix3 log_lst \
# --accuracy_type valid --output_plot acc_lst_sl_human.png

# python 1_plot_rf.py --log_folder1 ../condition1_human/training_log_context --log_prefix1 log_rf \
# --log_folder2 ../condition1_human/training_log_context --log_prefix2 log_rf \
# --log_folder3 ../condition1_human/training_log_context --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_comm_rf_human.png

# python 4_informativeness.py \
#   --input_files ../condition1_human/dump_context/msg_spk_seed222/sender_epoch29.txt \
#   --output_file ../condition1_human/dump_context/msg_spk_seed222/epochspk30_agent.txt \
#   > condition1_human_agent_sl_informativeness.txt

# python 4_informativeness.py \
#   --input_files ../condition1_human/dump_context/msg_rf_seed222/output_epoch30.txt \
#   --output_file ../condition1_human/dump_context/msg_rf_seed222/epoch30.txt \
#     --rl_or_sl > condition1_human_agent_rf_informativeness.txt
# ##### ##### ##### ##### ##### ##### #####  for test purpose ##### ##### ##### ##### ##### ##### 


# # ### Properties of the language
# # #### Color use 
# # #### Figure with frequency of use of each color in the test set. Three barplots side by side: one with human data, one after SL, one after RL1. Remember to sort them by human frequency.

# # ##### frequency of color use in the test set (human production (3000))
# # python 3_freq_human.py '../condition1_human/dump_context/msg_spk_seed111'
# # ##### frequency of color use in the test set (agent production after SL (3000))
# # python 3_freq_slrl.py '../condition1_human/dump_context/msg_spk_seed111' --specific_epoch 29
# # ##### frequency of color use in the test set (agent production after RL1 (3000))
# # python 3_freq_slrl.py '../condition1_human/dump_context/msg_rf_seed111' --specific_epoch 30


# # #### Figure with the evolution across epochs of the entropy of agents’ productions, measured on the test set. Single graph with two lines: one for SL, one for RL1. Put the human entropy as a horizontal line.

# # ##### [!] For SL train+test, it’s always 12434 train+3000 test human data.
# # ##### Test set is 3000 human data  (SL)
# # python 3_entropy_sl.py ../condition1_human/dump_context/msg_spk_seed111
# # python 3_entropy_sl.py ../condition1_human/dump_context/msg_spk_seed222
# # ##### Test set is 3000 human data  (RL1)
# python 3_entropy_rl_human.py ../condition1_human/dump_context/msg_rf_seed111
# python 3_entropy_rl_human.py ../condition1_human/dump_context/msg_rf_seed222


# # ##### Test set is generated data (SL) --- the human entropy should be the same 
# # python 3_entropy_sl.py ../condition3_generated/dump_context/msg_spk_seed222
# # python 3_entropy_sl.py ../condition3_generated/dump_context/msg_spk_seed111
# # ##### Test set is generated data (RL1)
# python 3_entropy_rl_generated.py ../condition3_generated/dump_context/msg_rf_seed111
# python 3_entropy_rl_generated.py ../condition3_generated/dump_context/msg_rf_seed222


# ### Informativeness
# #### Table for system-level informativeness, measured on the test set of the human data. Columns: humans, speaker-after-SL, speaker-after-RL. Rows are conditions: far, split, close, all. Add also one row for the difference between far and close. 


# ### Table for system-level informativeness, measured on the test set of the human data. 

# python 4_informativeness.py \
#     --input_files ../condition1_human/dump_context/msg_spk_seed111/sender_epoch29.txt \
#     --output_file ../condition1_human/dump_context/msg_spk_seed111/epochspk30.txt \
#     --use_label > condition1_human_label_informativeness.txt

# python 4_informativeness.py --input_files ../condition1_human/dump_context/msg_spk_seed111/sender_epoch29.txt \
#                          --output_file ../condition1_human/dump_context/msg_spk_seed111/epochspk30_agent.txt \
#                           > condition1_human_agent_sl_informativeness.txt

# python 4_informativeness.py --input_files ../condition1_human/dump_context/msg_rf_seed111/output_epoch30.txt \
#                   --output_file ../condition1_human/dump_context/msg_rf_seed111/epoch30.txt \
#                   --rl_or_sl > condition1_human_agent_rf_informativeness.txt


# for seed in 111 222 333 444 555 666 777 888 999
# do
#   python 4_informativeness.py \
#     --input_files ../condition1_human/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition1_human/dump_context/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> condition1_human_agent_rf_informativeness.txt
# done


# # 111, 555, 666, 888, 999


# #### Table for system-level informativeness, measured on the test set of the training set of the human data. 



# #### Table for system-level informativeness, measured on the test set of the generated data. 

# python 4_informativeness.py \
#   --input_files ../condition3_generated/condition_slrl3/dump_context/msg_spk_seed111/sender_epoch29.txt \
#   --output_file ../condition3_generated/condition_slrl3/dump_context/msg_spk_seed111/epochspk30_agent.txt \
#   > condition3_generated_condition_slrl3_agent_sl_informativeness.txt

# python 4_informativeness.py \
#   --input_files ../condition3_generated/condition_slrl3/dump_context/msg_spk_seed111/output_epoch30.txt \
#   --output_file ../condition3_generated/condition_slrl3/dump_context/msg_spk_seed111/epoch30.txt \
#     --rl_or_sl > condition3_generated_condition_slrl3_agent_rf_informativeness.txt


# for seed in 111 222 333 444 555 
# do
#   python 4_informativeness.py \
#     --input_files ../condition_allhuman/dump_context/msg_spk_seed${seed}/sender_epoch29.txt \
#     --output_file ../condition_allhuman/dump_context/msg_spk_seed${seed}/epochspk30_agent.txt \
#    >> condition_allhuman_agent_sl_informativeness.txt
# done

# for seed in 111 222 333 444 555 
# do
#   python 4_informativeness.py \
#     --input_files ../condition_allhuman/dump_context/msg_rf_seed${seed}/output_epoch0.txt \
#     --output_file ../condition_allhuman/dump_context/msg_rf_seed${seed}/epoch0.txt \
#     --rl_or_sl >> condition_allhuman_agent_rf_informativeness.txt
# done



for seed in 111 123 222
do
  python 4_informativeness.py \
    --input_files ../condition_allhuman/dump_context/msg_spk_seed${seed}/sender_epoch29.txt \
    --output_file ../condition_allhuman/dump_context/msg_spk_seed${seed}/epochspk30_agent.txt \
   >> condition_allhuman_agent_sl_informativeness.txt
done

for seed in 111 123 222
do
  python 4_informativeness.py \
    --input_files ../condition_allhuman/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
    --output_file ../condition_allhuman/dump_context/msg_rf_seed${seed}/epoch30.txt \
    --rl_or_sl >> condition_allhuman_agent_rf_informativeness.txt
done


# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl3/dump_context/msg_spk_seed${seed}/sender_epoch29.txt \
#     --output_file ../condition3_generated/condition_slrl3/dump_context/msg_spk_seed${seed}/epochspk30_agent.txt \
#    >> condition3_generated_condition_slrl3_agent_sl_informativeness.txt
# done

# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl3/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated/condition_slrl3/dump_context/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> condition3_generated_condition_slrl3_agent_rf_informativeness.txt
# done



# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl4/dump_context/msg_spk_seed${seed}/sender_epoch29.txt \
#     --output_file ../condition3_generated/condition_slrl4/dump_context/msg_spk_seed${seed}/epochspk30_agent.txt \
#    >> condition3_generated_condition_slrl4_agent_sl_informativeness.txt
# done

# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl4/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated/condition_slrl4/dump_context/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> condition3_generated_condition_slrl4_agent_rf_informativeness.txt
# done



# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl5/dump_context/msg_spk_seed${seed}/sender_epoch29.txt \
#     --output_file ../condition3_generated/condition_slrl5/dump_context/msg_spk_seed${seed}/epochspk30_agent.txt \
#    >> condition3_generated_condition_slrl5_agent_sl_informativeness.txt
# done

# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl5/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated/condition_slrl5/dump_context/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> condition3_generated_condition_slrl5_agent_rf_informativeness.txt
# done



# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl6/dump_context/msg_spk_seed${seed}/sender_epoch29.txt \
#     --output_file ../condition3_generated/condition_slrl6/dump_context/msg_spk_seed${seed}/epochspk30_agent.txt \
#    >> condition3_generated_condition_slrl6_agent_sl_informativeness.txt
# done

# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl6/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated/condition_slrl6/dump_context/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> condition3_generated_condition_slrl6_agent_rf_informativeness.txt
# done



# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl7/dump_context/msg_spk_seed${seed}/sender_epoch29.txt \
#     --output_file ../condition3_generated/condition_slrl7/dump_context/msg_spk_seed${seed}/epochspk30_agent.txt \
#    >> condition3_generated_condition_slrl7_agent_sl_informativeness.txt
# done

# for seed in 111 222 333 
# do
#   python 4_informativeness.py \
#     --input_files ../condition3_generated/condition_slrl7/dump_context/msg_rf_seed${seed}/output_epoch30.txt \
#     --output_file ../condition3_generated/condition_slrl7/dump_context/msg_rf_seed${seed}/epoch30.txt \
#     --rl_or_sl >> condition3_generated_condition_slrl7_agent_rf_informativeness.txt
# done


# #### prototypes

# for i in {0..30}; do \
#     python 2_prototype.py ../condition3_generated/test/dump_context/msg_rf_seed222/output_epoch${i}.txt \
#     --pickle_output prototypes_rf_human_epoch${i}.pkl \
#     --csv_output prototypes_rf_human_epoch${i}.csv --rl; \
# done

# for i in {0..30}; do \
#     python 2_prototype.py ../condition3_generated/test/dump/msg_rf_seed222/output_epoch${i}.txt \
#     --pickle_output prototypes_rf_human_epoch${i}.pkl \
#     --csv_output prototypes_rf_human_epoch${i}.csv --rl; \
# done

# for i in {0..30}; do \
#     python 2_prototype.py ../condition3_generated/test/dump_exp/msg_rf_seed222/output_epoch${i}.txt \
#     --pickle_output prototypes_rf_human_epoch${i}.pkl \
#     --csv_output prototypes_rf_human_epoch${i}.csv --rl; \
# done





# for i in {0..29}; do \
#     python 2_prototype.py ../condition3_generated/condition_slrl3/dump_context/msg_spk_seed111/sender_epoch${i}.txt \
#     --pickle_output prototypes_spk_human_epoch${i}.pkl \
#     --csv_output prototypes_spk_human_epoch${i}.csv; \
# done

# python 2_prototype.py ../condition3_generated/condition_slrl3/dump_context/msg_spk_seed111/sender_epoch29.txt \
#     --pickle_output prototypes_spk_label.pkl \
#     --csv_output prototypes_spk_label.csv \
#     --use_label


# python 2_prototype.py ../condition1_human/dump_context/msg_rf_seed111/output_epoch30.txt \
#     --pickle_output prototypes_rf_human.pkl \
#     --csv_output prototypes_rf_human.csv \
#     --rl

# python 2_prototype.py ../condition1_human/dump_context/msg_spk_seed111/sender_epoch29.txt \
#     --pickle_output prototypes_spk_human.pkl \
#     --csv_output prototypes_spk_human.csv 

# python 2_prototype.py ../condition1_human/dump_context/msg_spk_seed111/sender_epoch29.txt \
#     --pickle_output prototypes_spk_label.pkl \
#     --csv_output prototypes_spk_label.csv \
#     --use_label


# python 5_typicality.py --mode model --data_file ../condition1_human/dump_context/msg_rf_seed111/output_epoch30.txt --prototypes_file prototypes_rf_human.pkl --output_csv typicality_results_rf_human --rl_or_sl 
# python 5_typicality.py --mode model --data_file ../condition1_human/dump_context/msg_spk_seed111/sender_epoch29.txt --prototypes_file prototypes_spk_human.pkl --output_csv typicality_results_spk_human
# python 5_typicality.py --mode model --data_file ../condition1_human/dump_context/msg_spk_seed111/sender_epoch29.txt --prototypes_file prototypes_spk_label.pkl --output_csv typicality_results_spk_label

# python 5_typicality.py --mode model --data_file ./condition_allhuman.csv --prototypes_file ../dynamics/language_use/prototypes.pkl --output_csv typicality_results_allhuman


# python 5_typicality.py --mode human --data_file ../dynamics/language_use/successful_trials_cielab.csv --prototypes_file ../dynamics/language_use/prototypes.pkl --output_csv typicality_results_human





# python plot_sl.py --log_folder1 ./training_log --log_prefix1 log_spk \
# --log_folder2 ./training_log_context --log_prefix2 log_spk \
# --log_folder3 ./training_log_exp --log_prefix3 log_spk \
# --accuracy_type valid --output_plot acc_spk_valid.png

# python plot_sl.py --log_folder1 ./training_log --log_prefix1 log_spk \
# --log_folder2 ./training_log_context --log_prefix2 log_spk \
# --log_folder3 ./training_log_exp --log_prefix3 log_spk \
# --accuracy_type far --output_plot acc_spk_far.png

# python plot_sl.py --log_folder1 ./training_log --log_prefix1 log_spk \
# --log_folder2 ./training_log_context --log_prefix2 log_spk \
# --log_folder3 ./training_log_exp --log_prefix3 log_spk \
# --accuracy_type close --output_plot acc_spk_close.png

# python plot_sl.py --log_folder1 ./training_log --log_prefix1 log_spk \
# --log_folder2 ./training_log_context --log_prefix2 log_spk \
# --log_folder3 ./training_log_exp --log_prefix3 log_spk \
# --accuracy_type split --output_plot acc_spk_split.png



# python plot_sl.py --log_folder1 ./training_log --log_prefix1 log_lst \
# --log_folder2 ./training_log_context --log_prefix2 log_lst \
# --log_folder3 ./training_log_exp --log_prefix3 log_lst \
# --accuracy_type valid --output_plot acc_lst_valid.png

# python plot_sl.py --log_folder1 ./training_log --log_prefix1 log_lst \
# --log_folder2 ./training_log_context --log_prefix2 log_lst \
# --log_folder3 ./training_log_exp --log_prefix3 log_lst \
# --accuracy_type far --output_plot acc_lst_far.png

# python plot_sl.py --log_folder1 ./training_log --log_prefix1 log_lst \
# --log_folder2 ./training_log_context --log_prefix2 log_lst \
# --log_folder3 ./training_log_exp --log_prefix3 log_lst \
# --accuracy_type close --output_plot acc_lst_close.png

# python plot_sl.py --log_folder1 ./training_log --log_prefix1 log_lst \
# --log_folder2 ./training_log_context --log_prefix2 log_lst \
# --log_folder3 ./training_log_exp --log_prefix3 log_lst \
# --accuracy_type split --output_plot acc_lst_split.png


# python plot_rf.py --log_folder1 ./training_log --log_prefix1 log_rf \
# --log_folder2 ./training_log_context --log_prefix2 log_rf \
# --log_folder3 ./training_log_exp --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_rf_valid.png

# python plot_rf.py --log_folder1 ./training_log --log_prefix1 log_rf \
# --log_folder2 ./training_log_context --log_prefix2 log_rf \
# --log_folder3 ./training_log_exp --log_prefix3 log_rf \
# --accuracy_type far --output_plot acc_rf_far.png

# python plot_rf.py --log_folder1 ./training_log --log_prefix1 log_rf \
# --log_folder2 ./training_log_context --log_prefix2 log_rf \
# --log_folder3 ./training_log_exp --log_prefix3 log_rf \
# --accuracy_type close --output_plot acc_rf_close.png

# python plot_rf.py --log_folder1 ./training_log --log_prefix1 log_rf \
# --log_folder2 ./training_log_context --log_prefix2 log_rf \
# --log_folder3 ./training_log_exp --log_prefix3 log_rf \
# --accuracy_type split --output_plot acc_rf_split.png


# ### plot rf spk

# python plot_rf_spk.py --log_folder1 ./training_log --log_prefix1 log_rf \
# --log_folder2 ./training_log_context --log_prefix2 log_rf \
# --log_folder3 ./training_log_exp --log_prefix3 log_rf \
# --accuracy_type valid --output_plot acc_rf_spk_valid.png

# python plot_rf_spk.py --log_folder1 ./training_log --log_prefix1 log_rf \
# --log_folder2 ./training_log_context --log_prefix2 log_rf \
# --log_folder3 ./training_log_exp --log_prefix3 log_rf \
# --accuracy_type far --output_plot acc_rf_spk_far.png

# python plot_rf_spk.py --log_folder1 ./training_log --log_prefix1 log_rf \
# --log_folder2 ./training_log_context --log_prefix2 log_rf \
# --log_folder3 ./training_log_exp --log_prefix3 log_rf \
# --accuracy_type close --output_plot acc_rf_spk_close.png

# python plot_rf_spk.py --log_folder1 ./training_log --log_prefix1 log_rf \
# --log_folder2 ./training_log_context --log_prefix2 log_rf \
# --log_folder3 ./training_log_exp --log_prefix3 log_rf \
# --accuracy_type split --output_plot acc_rf_spk_split.png


