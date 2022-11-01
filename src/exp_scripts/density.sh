EXP_NAME=density_11-1

#set -x
#trap read debug
#
python experiment_generator.py -cfg exp_cfg/density_tests/density_40.toml -o "${EXP_NAME}_40_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_10.toml -o "${EXP_NAME}_10_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_20.toml -o "${EXP_NAME}_20_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_30.toml -o "${EXP_NAME}_30_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_40.toml -o "${EXP_NAME}_40_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_50.toml -o "${EXP_NAME}_50_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_60.toml -o "${EXP_NAME}_60_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_70.toml -o "${EXP_NAME}_70_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_80.toml -o "${EXP_NAME}_80_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_90.toml -o "${EXP_NAME}_90_exp"
python experiment_generator.py -cfg exp_cfg/density_tests/density_100.toml -o "${EXP_NAME}_100_exp"




# experiment 1 -- limited domain
python experiments_to_json.py -d "experiment_data/${EXP_NAME}_06_exp_0/" \
                              "experiment_data/${EXP_NAME}_08_exp_0/" \
                              "experiment_data/${EXP_NAME}_10_exp_0/" \
                              "experiment_data/${EXP_NAME}_15_exp_0/" \
                              "experiment_data/${EXP_NAME}_20_exp_0/" \
                              "experiment_data/${EXP_NAME}_25_exp_0/" \
                              -o "all_data_${EXP_NAME}.json"

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
