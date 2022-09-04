EXP_NAME=makespan_9_3

#python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_02.toml
#python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_04.toml
#python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_06.toml
#python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_08.toml
#python experiment_generator.py -cfg exp_cfg/makespan_tests/makespan_10.toml

python experiments_to_json.py -d 'experiment_data/makespan_02_exp_0/' \
                              'experiment_data/makespan_04_exp_0/' \
                              'experiment_data/makespan_06_exp_0/' \
                              'experiment_data/makespan_08_exp_0/' \
                              'experiment_data/makespan_10_exp_0/' \
                              -o "all_data_${EXP_NAME}.json"

python graph_json.py -f "all_data_${EXP_NAME}.json"
