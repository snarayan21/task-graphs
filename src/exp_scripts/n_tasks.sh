EXP_NAME=tasks_9_12

#set -x
#trap read debug
#
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_06.toml -o "${EXP_NAME}_06_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_08.toml -o "${EXP_NAME}_08_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_10.toml -o "${EXP_NAME}_10_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_15.toml -o "${EXP_NAME}_15_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_20.toml -o "${EXP_NAME}_20_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_25.toml -o "${EXP_NAME}_25_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_30.toml -o "${EXP_NAME}_30_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_35.toml -o "${EXP_NAME}_35_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_40.toml -o "${EXP_NAME}_40_exp"

# experiment 1 -- limited domain
python experiments_to_json.py -d "experiment_data/${EXP_NAME}_06_exp_0/" \
                              "experiment_data/${EXP_NAME}_08_exp_0/" \
                              "experiment_data/${EXP_NAME}_10_exp_0/" \
                              "experiment_data/${EXP_NAME}_15_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_exp_0/" \
                              -o "all_data_${EXP_NAME}_lim.json"

# experiment 1 -- full domain
 python experiments_to_json.py -d "experiment_data/${EXP_NAME}_06_exp_0/" \
                              "experiment_data/${EXP_NAME}_08_exp_0/" \
                              "experiment_data/${EXP_NAME}_10_exp_0/" \
                              "experiment_data/${EXP_NAME}_15_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_exp_0/" \
                              "experiment_data/${EXP_NAME}_25_exp_0/" \
                              "experiment_data/${EXP_NAME}_30_exp_0/" \
                              "experiment_data/${EXP_NAME}_35_exp_1/" \
                              "experiment_data/${EXP_NAME}_40_exp_0/" \
                              -o "all_data_${EXP_NAME}.json"


python graph_json.py -f "all_data_${EXP_NAME}_lim.json" -g n_tasks
python graph_json.py -f "all_data_${EXP_NAME}.json" -g n_tasks
