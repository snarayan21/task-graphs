EXP_NAME=agents_9_5

#set -x
#trap read debug

python experiment_generator.py -cfg exp_cfg/nagents_tests/agents_02.toml -o "${EXP_NAME}_02_exp"
python experiment_generator.py -cfg exp_cfg/nagents_tests/agents_04.toml -o "${EXP_NAME}_04_exp"
python experiment_generator.py -cfg exp_cfg/nagents_tests/agents_06.toml -o "${EXP_NAME}_06_exp"
python experiment_generator.py -cfg exp_cfg/nagents_tests/agents_10.toml -o "${EXP_NAME}_10_exp"

python experiments_to_json.py -d "experiment_data/${EXP_NAME}_02_exp_0/" \
                              "experiment_data/${EXP_NAME}_04_exp_1/" \
                              "experiment_data/${EXP_NAME}_06_exp_0/" \
                              "experiment_data/${EXP_NAME}_10_exp_0/" \
                              -o "all_data_${EXP_NAME}.json"

python graph_json.py -f "all_data_${EXP_NAME}.json" -g n_agents
