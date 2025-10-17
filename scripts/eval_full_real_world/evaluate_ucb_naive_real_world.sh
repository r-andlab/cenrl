#!bin/bash

# Make sure $HOME points to the parent directory of cenrl
# EXAMPLE TEST RUN using ucb_naive_api_test: HOME="/path/to/home" bash scripts/eval_full_real_world/evaluate_ucb_naive_real_world.sh -e 1 -m 1000 -f "categories" -s "top10k" -a "ucb_naive_api_test" -r "false"
# EXAMPLE RUN ucb_naive_api: HOME="/path/to/home" bash scripts/eval_full_real_world/evaluate_ucb_naive_real_world.sh -e 1 -m 1000 -f "categories" -s "top10k" -a "ucb_naive_api" -r "false"
RLFOLDER=$HOME/cenrl
timestamp=$(date +%Y%m%d_%H%M%S)

#Get some inputs
while getopts ":e:m:g:t:f:s:a:d:p:r:" option; do
  case $option in
    e)
      episodes=$OPTARG      # episodes
      ;;
    m)
      measurements=$OPTARG    # measurements
      ;;
    g)
      ground_truth_option=$OPTARG   #GT option
      ;;
    t)
      selected_target_max_try_option=$OPTARG
      ;;
    f)
      features=(${OPTARG})
      ;;
    s)
      size=$OPTARG  
      ;;
    a)
      algorithm=$OPTARG
      ;;
    d)
      per_date_threshold=$OPTARG
      ;;
    p)
      action_space_multi_parents=$([[ "${OPTARG:-false}" == "true" ]] && echo "-asp" || echo "")
      ;;
    r)
      sample_by_target_rank=$([[ "${OPTARG:-false}" == "true" ]] && echo "-sr" || echo "")
      ;;
    \?)
      echo "Invalid option -$OPTARG"
      ;;
  esac
done

selected_target_max_try=${selected_target_max_try_option:-5}

# finding the given model
modelpathbase=$RLFOLDER/api/
modelpath=$(find $modelpathbase -type f -name "${algorithm}.py")

# create new output dir
outputdir=$RLFOLDER/real-world/outputs/${algorithm}_${size}_${ground_truth_option}_$(printf "%s_" "${features[@]}")${timestamp}
mkdir -p ${outputdir}

ground_truth=$RLFOLDER/inputs/${ground_truth_option}/${ground_truth_option}-blocklist.csv
tranco_file=$RLFOLDER/inputs/tranco/tranco_categories_subdomain_tld_entities_${size}.csv

#Print out stuff
echo "Using model name: ${algorithm}"
if [[ -n "$ground_truth_option" ]]; then
  echo "Using ground truth: ${ground_truth_option}"
  echo "Using ground truth file: ${ground_truth}"
fi
echo "Using features: ${features[@]}"
echo "Using action_space_multi_parents: ${action_space_multi_parents}"
echo "Using per_date_threshold: ${per_date_threshold}"
echo "Using selected_target_max_try: ${selected_target_max_try}"
echo "Using size: ${size}"
echo "Using action space file: ${tranco_file}"
echo "Using sample by target rank: ${sample_by_target_rank}"

# run the field model with hyperparams, change them as necessary
cd $RLFOLDER/api
ucb_c=0.03
stepsize=0.0
initq=0.0
python3 $modelpath $sample_by_target_rank -m $measurements -E $episodes -c=$ucb_c -s $stepsize -V $initq -o ${outputdir}/${algorithm}_c${ucb_c}_stepsize${stepsize}_initval${initq} -g ${ground_truth} -a ${tranco_file} -f ${features[@]} &


# plot the results
wait
cd $RLFOLDER/scripts
python3 plotter_dyn.py --output_file_name ${algorithm}_grid.pdf --results_directory $outputdir --results_prefix $algorithm --ground_truth_file_path $ground_truth --measurements $measurements --episodes $episodes --action_space_file_path $tranco_file

echo "Done"
