#!bin/bash

# Example command line
# HOME="/path/to/home" bash scripts/eval_full/evaluate_baselines.sh -e 1 -m 10000 -g gfwatch -s "top10k" -d 100 -t 1

RLFOLDER=$HOME/cenrl
timestamp=$(date +%Y%m%d_%H%M%S)

#Get some inputs
while getopts ":e:m:g:t:s:d:" option; do
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
    s)
      size=$OPTARG    #Size can be either top10k or top1m
      ;;
    d)
      per_date_threshold=$OPTARG
      ;;
    \?)
      echo "Invalid option -$OPTARG"
      ;;
  esac
done

# used for baselines with replacement
selected_target_max_try=${selected_target_max_try_option:-1}

outputdir=$RLFOLDER/models/outputs/baselines_10k_${ground_truth_option}_${timestamp}
mkdir -p ${outputdir}
ground_truth=$RLFOLDER/inputs/${ground_truth_option}/${ground_truth_option}-blocklist.csv
tranco_file=$RLFOLDER/inputs/tranco/tranco_categories_subdomain_tld_entities_${size}.csv

echo "Using ground truth: ${ground_truth_option}"
echo "Using ground truth file: ${ground_truth}"
echo "Using per_date_threshold: ${per_date_threshold}"
echo "Using selected_target_max_try: ${selected_target_max_try}"
echo "Using size: ${size}"
echo "Using action space file: ${tranco_file}"



cd $RLFOLDER/scripts
python3 plotter_baselines_only.py --output_file_name baselines_grid.pdf --output_directory $outputdir --ground_truth_file_path $ground_truth --measurements $measurements --episodes $episodes --action_space_file_path $tranco_file --per_date_threshold $per_date_threshold --selected_target_max_try $selected_target_max_try
