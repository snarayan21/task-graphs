EXP_NAME=heatmap_9_12_600t


python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_08_makespan_50.toml -o "${EXP_NAME}_08_50_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_12_makespan_50.toml -o "${EXP_NAME}_12_50_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_16_makespan_50.toml -o "${EXP_NAME}_16_50_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_20_makespan_50.toml -o "${EXP_NAME}_20_50_exp"

python experiments_to_json.py -d "experiment_data/${EXP_NAME}_08_50_exp_0/" \
                              "experiment_data/${EXP_NAME}_12_50_exp_0/" \
                              "experiment_data/${EXP_NAME}_16_50_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_50_exp_0/" \
                              -o "all_data_${EXP_NAME}_50.json"

python graph_json.py -f "all_data_${EXP_NAME}_50.json" -g n_tasks
