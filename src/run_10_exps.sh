# Concatenate the argument with a file name string
file_name="exp_cfg/ntasks_tests/$1"

for ((i=1; i<=10; i++)); do
    python run_realtime_experiment.py -cfg $file_name -o $2
done
