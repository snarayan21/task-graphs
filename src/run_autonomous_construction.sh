declare -a TaskNumberArray=("8" "12" "16" "20")
declare -a TrialNumberArray=("1" "2" "3" "4" "5")

for tasknumber in ${TaskNumberArray[@]}
do
  for trialnumber in ${TrialNumberArray[@]}
  do
    python experiment_generator.py -cfg exp_cfg/from_file.toml -inputargs autonomous_construction/experiments_examples/${tasknumber}nodes_${trialnumber}.toml
  done
done
