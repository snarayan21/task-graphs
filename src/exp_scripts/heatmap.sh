EXP_NAME=heatmap_9_9

#set -x
#trap read debug

python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_08_makespan_25.toml -o "${EXP_NAME}_08_25_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_08_makespan_50.toml -o "${EXP_NAME}_08_50_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_08_makespan_75.toml -o "${EXP_NAME}_08_75_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_08_makespan_10.toml -o "${EXP_NAME}_08_10_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_12_makespan_25.toml -o "${EXP_NAME}_12_25_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_12_makespan_50.toml -o "${EXP_NAME}_12_50_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_12_makespan_75.toml -o "${EXP_NAME}_12_75_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_12_makespan_10.toml -o "${EXP_NAME}_12_10_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_16_makespan_25.toml -o "${EXP_NAME}_16_25_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_16_makespan_50.toml -o "${EXP_NAME}_16_50_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_16_makespan_75.toml -o "${EXP_NAME}_16_75_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_16_makespan_10.toml -o "${EXP_NAME}_16_10_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_20_makespan_25.toml -o "${EXP_NAME}_20_25_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_20_makespan_50.toml -o "${EXP_NAME}_20_50_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_20_makespan_75.toml -o "${EXP_NAME}_20_75_exp"
python experiment_generator.py -cfg exp_cfg/ntasks_tests/tasks_20_makespan_10.toml -o "${EXP_NAME}_20_10_exp"



python experiments_to_json.py -d "experiment_data/${EXP_NAME}_08_25_exp_0/" \
                              "experiment_data/${EXP_NAME}_12_25_exp_0/" \
                              "experiment_data/${EXP_NAME}_16_25_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_25_exp_0/" \
                              -o "all_data_${EXP_NAME}_25.json"

python experiments_to_json.py -d "experiment_data/${EXP_NAME}_08_50_exp_0/" \
                              "experiment_data/${EXP_NAME}_12_50_exp_0/" \
                              "experiment_data/${EXP_NAME}_16_50_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_50_exp_0/" \
                              -o "all_data_${EXP_NAME}_50.json"

python experiments_to_json.py -d "experiment_data/${EXP_NAME}_08_75_exp_0/" \
                              "experiment_data/${EXP_NAME}_12_75_exp_0/" \
                              "experiment_data/${EXP_NAME}_16_75_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_75_exp_0/" \
                              -o "all_data_${EXP_NAME}_75.json"

python experiments_to_json.py -d "experiment_data/${EXP_NAME}_08_10_exp_0/" \
                              "experiment_data/${EXP_NAME}_12_10_exp_0/" \
                              "experiment_data/${EXP_NAME}_16_10_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_10_exp_0/" \
                              -o "all_data_${EXP_NAME}_10.json"

python graph_json.py -f "all_data_${EXP_NAME}.json" -g n_tasks
