EXP_NAME=makespan_9_9_mix_600t

#set -x
#trap read debug

python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_02.toml -o "${EXP_NAME}_02_exp"
python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_04.toml -o "${EXP_NAME}_04_exp"
python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_06.toml -o "${EXP_NAME}_06_exp"
python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_08.toml -o "${EXP_NAME}_08_exp"
python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_10.toml -o "${EXP_NAME}_10_exp"

python experiments_to_json.py -d "experiment_data/${EXP_NAME}_02_exp_0/" \
                              "experiment_data/${EXP_NAME}_04_exp_0/" \
                              "experiment_data/${EXP_NAME}_06_exp_0/" \
                              "experiment_data/${EXP_NAME}_08_exp_0/" \
                              "experiment_data/${EXP_NAME}_10_exp_0/" \
                              -o "all_data_${EXP_NAME}.json"

python graph_json.py -f "all_data_${EXP_NAME}.json" -g makespan
