EXP_NAME=heatmap_9_11_600t


python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_08_makespan_25.toml -o "${EXP_NAME}_08_25_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_12_makespan_25.toml -o "${EXP_NAME}_12_25_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_16_makespan_25.toml -o "${EXP_NAME}_16_25_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_20_makespan_25.toml -o "${EXP_NAME}_20_25_exp"

python experiments_to_json.py -d "experiment_data/${EXP_NAME}_08_25_exp_0/" \
                              "experiment_data/${EXP_NAME}_12_25_exp_0/" \
                              "experiment_data/${EXP_NAME}_16_25_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_25_exp_0/" \
                              -o "all_data_${EXP_NAME}_25.json"

python graph_json.py -f "all_data_${EXP_NAME}_25.json" -g n_tasks
