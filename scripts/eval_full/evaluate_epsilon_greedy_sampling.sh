#!bin/bash

# Make sure $HOME points to the parent directory of cenrl
#EXAMPLE TO RUN: HOME="/path/to/home" bash scripts/eval_full/evaluate_epsilon_greedy_sampling.sh -e 2 -m 1000 -g gfwatch -p "true" -f "categories" -s "top10k" -a "epsilon_greedy_sampling -r "false"

RLFOLDER=$HOME/cenrl
timestamp=$(date +%Y%m%d_%H%M%S)
num_of_processes_for_episodes=1

#Get some inputs
while getopts ":e:m:g:t:f:s:a:d:p:r:b:n:" option; do
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
      size=$OPTARG    #Size can be either top10k or top1m
      ;;
    a)
      algorithm=$OPTARG
      ;;
    d)
      per_date_threshold=$OPTARG
      ;;
    n)
      num_of_processes_for_episodes=$OPTARG
      ;;
    p)
      # use -p "true" to set
      action_space_multi_parents=$([[ "${OPTARG:-false}" == "true" ]] && echo "-asp" || echo "")
      ;;
    r)
      # use -r "true" to set
      sample_by_target_rank=$([[ "${OPTARG:-false}" == "true" ]] && echo "-sr" || echo "")
      ;;
    b)
      # use -b "true" to set
      reset_more_per_date_threshold=$([[ "${OPTARG:-false}" == "true" ]] && echo "-rmpdt" || echo "")
      ;;
    \?)
      echo "Invalid option -$OPTARG"
      ;;
  esac
done

selected_target_max_try=${selected_target_max_try_option:-5}

modelpathbase=$RLFOLDER/models/
modelpath=$(find $modelpathbase -type f -name "${algorithm}.py")

if [[ $algorithm == *"_dyn_ordered"* ]]; then
  longitudinal="dyn_ordered"
elif [[ $algorithm == *"_dyn_blocklists"* ]]; then
  longitudinal="dyn_blocklists"
else
  longitudinal="static"
fi



outputdir=$RLFOLDER/models/outputs/${algorithm}_${size}_${ground_truth_option}_$(printf "%s_" "${features[@]}")${timestamp}
mkdir -p ${outputdir}
if [[ $longitudinal == *"dyn_"* ]]; then
  ground_truth=$RLFOLDER/inputs/${ground_truth_option}/${ground_truth_option}-longitudinal-blocklist.csv
else
  ground_truth=$RLFOLDER/inputs/${ground_truth_option}/${ground_truth_option}-blocklist.csv
fi

tranco_file=$RLFOLDER/inputs/tranco/tranco_categories_subdomain_tld_entities_${size}.csv


echo "Using model name: ${algorithm}"
echo "Using ground truth: ${ground_truth_option}"
echo "Using ground truth file: ${ground_truth}"
echo "Using features: ${features[@]}"
echo "Using action_space_multi_parents: ${action_space_multi_parents}"
echo "Using per_date_threshold: ${per_date_threshold}"
echo "Using reset_more_per_date_threshold: ${reset_more_per_date_threshold}"
echo "Using selected_target_max_try: ${selected_target_max_try}"
echo "Using size: ${size}"
echo "Using action space file: ${tranco_file}"
echo "Using sample by target rank: ${sample_by_target_rank}"
echo "Using num_of_processes_for_episodes: ${num_of_processes_for_episodes}"


cd $RLFOLDER/models

# Nested for loop
for epsilon in 0.2 0.4 0.6 0.8
do
  # epsilon loop
  for stepsize in 0.0 0.2 0.4 0.6 0.8
  do
    # stepsize loop
    for initq in 0.0 0.2 0.4 0.6 0.8
    do
      if [[ $longitudinal == "static" ]]; then
        python3 $modelpath $sample_by_target_rank -np $num_of_processes_for_episodes -m $measurements -E $episodes -e $epsilon -s $stepsize -V $initq -o ${outputdir}/${algorithm}_eps${epsilon}_stepsize${stepsize}_initval${initq} -g ${ground_truth} -a ${tranco_file} -f ${features[@]} 2>&1 &
      elif [[ $longitudinal == "dyn_blocklists" ]]; then
        python3 $modelpath $sample_by_target_rank -np $num_of_processes_for_episodes $action_space_multi_parents -m $measurements -E $episodes -e $epsilon -s $stepsize -V $initq -o ${outputdir}/${algorithm}_eps${epsilon}_stepsize${stepsize}_initval${initq} -g ${ground_truth} -a ${tranco_file} -f ${features[@]} --selected_target_max_try ${selected_target_max_try} 2>&1 &
      elif [[ $longitudinal == "dyn_ordered" ]]; then
        python3 $modelpath $sample_by_target_rank -np $num_of_processes_for_episodes $reset_more_per_date_threshold $action_space_multi_parents -m $measurements -E $episodes -e $epsilon -s $stepsize -V $initq -o ${outputdir}/${algorithm}_eps${epsilon}_stepsize${stepsize}_initval${initq} -g ${ground_truth} -a ${tranco_file} -f ${features[@]} -pdt ${per_date_threshold} --selected_target_max_try ${selected_target_max_try} 2>&1 &
      else
        echo "Invalid long"
      fi
    done
    echo "running with (epsilon=${epsilon},stepsize=${stepsize})"
  done
done

wait
cd $RLFOLDER/scripts
if [[ $longitudinal == "static" ]]; then
  python3 plotter_dyn.py --output_file_name ${algorithm}_grid.pdf --results_directory $outputdir --results_prefix $algorithm --ground_truth_file_path $ground_truth --measurements $measurements --episodes $episodes --action_space_file_path $tranco_file
elif [[ $longitudinal == "dyn_blocklists" ]]; then
  python3 plotter_dyn.py --output_file_name ${algorithm}_grid.pdf --results_directory $outputdir --results_prefix $algorithm --ground_truth_file_path $ground_truth --measurements $measurements --episodes $episodes --action_space_file_path $tranco_file --selected_target_max_try $selected_target_max_try
elif [[ $longitudinal == "dyn_ordered" ]]; then
  python3 plotter_dyn.py --output_file_name ${algorithm}_grid.pdf --results_directory $outputdir --results_prefix $algorithm --ground_truth_file_path $ground_truth --measurements $measurements --episodes $episodes --action_space_file_path $tranco_file --selected_target_max_try_ordered $selected_target_max_try --per_date_threshold $per_date_threshold
else
  echo "Invalid long"
fi

echo "Done with plotting"
