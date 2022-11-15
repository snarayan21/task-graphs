EXP_NAME=longest_path_11-8

#set -x
#trap read debug
#
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_1.toml -o "${EXP_NAME}_1_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_2.toml -o "${EXP_NAME}_2_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_3.toml -o "${EXP_NAME}_3_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_4.toml -o "${EXP_NAME}_4_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_5.toml -o "${EXP_NAME}_5_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_6.toml -o "${EXP_NAME}_6_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_7.toml -o "${EXP_NAME}_7_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_8.toml -o "${EXP_NAME}_8_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_9.toml -o "${EXP_NAME}_9_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_10.toml -o "${EXP_NAME}_10_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_11.toml -o "${EXP_NAME}_11_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_12.toml -o "${EXP_NAME}_12_exp"
python experiment_generator.py -cfg exp_cfg/longest_path_tests/path_13.toml -o "${EXP_NAME}_13_exp"




python experiments_to_json.py -d "experiment_data/${EXP_NAME}_1_exp_0/" \
                              "experiment_data/${EXP_NAME}_2_exp_0/" \
                              "experiment_data/${EXP_NAME}_3_exp_0/" \
                              "experiment_data/${EXP_NAME}_4_exp_0/" \
                              "experiment_data/${EXP_NAME}_5_exp_0/" \
                              "experiment_data/${EXP_NAME}_6_exp_0/" \
                              "experiment_data/${EXP_NAME}_7_exp_0/" \
                              "experiment_data/${EXP_NAME}_8_exp_0/" \
                              "experiment_data/${EXP_NAME}_9_exp_0/" \
                              "experiment_data/${EXP_NAME}_10_exp_0/" \
                              "experiment_data/${EXP_NAME}_11_exp_0/" \
                              "experiment_data/${EXP_NAME}_12_exp_0/" \
                              "experiment_data/${EXP_NAME}_13_exp_0/" \
                              -o "all_data_${EXP_NAME}.json"




python graph_json.py -f "all_data_${EXP_NAME}.json" -g longest_path
