declare -a TaskNumberArray=("8") # "12" "16" "20")
declare -a TrialNumberArray=("1" "2" "3" "4" "5")

for tasknumber in ${TaskNumberArray[@]}
do
  for trialnumber in ${TrialNumberArray[@]}
  do
    python experiment_generator.py -cfg exp_cfg/from_file.toml -inputargs autonomous_construction/experiments_examples/${tasknumber}nodes_${trialnumber}.toml
  done
done
#
#python experiments_to_json.py -d "experiment_data/autonomous_construction_experiment_data/8nodes" \
#                              "experiment_data/autonomous_construction_experiment_data/12nodes" \
#                              "experiment_data/autonomous_construction_experiment_data/16nodes" \
#                              "experiment_data/autonomous_construction_experiment_data/20nodes" \
#                              -o "all_data_autonomous_construction.json"
#
#python graph_json.py -f "all_data_autonomous_construction.json" -g n_tasks
