EXP_NAME=tasks_9_5

#set -x
#trap read debug

#python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_06.toml -o "${EXP_NAME}_06_exp"
#python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_08.toml -o "${EXP_NAME}_08_exp"
#python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_10.toml -o "${EXP_NAME}_10_exp"
#python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_15.toml -o "${EXP_NAME}_15_exp"

python experiments_to_json.py -d "experiment_data/${EXP_NAME}_06_exp_0/" \
                              "experiment_data/${EXP_NAME}_08_exp_0/" \
                              "experiment_data/${EXP_NAME}_10_exp_0/" \
                              "experiment_data/${EXP_NAME}_15_exp_0/" \
                              -o "all_data_${EXP_NAME}.json"

python graph_json.py -f "all_data_${EXP_NAME}.json" -g n_tasks
