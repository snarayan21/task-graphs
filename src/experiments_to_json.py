import toml
import json
import argparse
import pathlib

def main():

    parser = argparse.ArgumentParser(description='Generate json file from experiments.')
    parser.add_argument('-d','--dirlist', nargs='+', help='<Required> List of experiment directories', required=True)
    args = parser.parse_args()

    directories = [pathlib.Path(directory) for directory in args.dirlist]

    json_output = {}
    for path in directories:
        json_all_trials = {}
        trialnum = 0
        for trial_folder in path.iterdir():
            json_trial = {}
            if trial_folder.is_dir():
                print((trial_folder / 'args.toml').absolute())
                json_trial["args"] = toml.load((trial_folder / 'args.toml').absolute())
                json_trial["results"] = toml.load((trial_folder / 'results.toml').absolute())
                json_all_trials["trial " + str(trialnum)] = json_trial
                trialnum += 1
        json_output[str(path)] = json_all_trials
    breakpoint()
    with open("all_experiments.json", "w") as outfile:
        json.dump(json_output, outfile)

if __name__ == '__main__':
    main()
