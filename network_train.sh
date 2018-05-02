# Create temporary starage of file not to get overwirten if batch job dosn't start immediately
tmpfile=$(mktemp ~/tmp/params.XXXXXX)
cat $1 > $tmpfile
export TEMP_PARAMS_FILE=$tmpfile

if [[ $2 == "test" ]]; then
  sh network_train.job
else
  ID=$(python -c "import ast; from os.path import normpath, basename; f = open('$TEMP_PARAMS_FILE', 'r'); p = ast.literal_eval(str(f.read())); print(basename(normpath(p['log_root_dir'])))")
  ntasks_per_node=$(python -c "import ast; f = open('$TEMP_PARAMS_FILE', 'r'); p = ast.literal_eval(str(f.read())); print(p['ntasks_per_node'])")
  echo "Using ntasks-per-node="$ntasks_per_node
  sbatch --job-name "$ID" --ntasks-per-node $ntasks_per_node network_train.job
fi
