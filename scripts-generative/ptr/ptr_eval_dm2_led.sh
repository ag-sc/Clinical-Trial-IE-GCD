#!/bin/bash
cd "$(dirname "$0")"
cd ..
cd ..
source venv/bin/activate
export PYTHONPATH=./src/
configdate=$(date '+%Y-%m-%d-%H:%M:%S')
echo 'allenai/led-base-16384' 'google/flan-t5-base'

modelname='allenai/led-base-16384'
mnshort=$(echo "$modelname" | sed -E 's/.*?\/(.*)/\1/g')
echo $modelname $mnshort

if [ -z $1 ]; then
  configdate=$(date '+%Y-%m-%d-%H:%M:%S')
  conffile=config_gen_dm2_"$mnshort"_"$configdate"_basic.json
  python -u src/generative_approach/create_task_config_file.py --host local --disease-prefix dm2 --min-slot-freq 10 --filename $conffile --batch-size 1 --epochs 50 --model $modelname
  python -u src/generative_approach/training.py $conffile | tee $conffile.log
else
  configdate=$(date '+%Y-%m-%d-%H:%M:%S')
  conffile=config_gen_dm2_"$mnshort"_"$configdate"_ptrmodel.json
  python -u src/generative_approach/create_task_config_file.py --host local --disease-prefix dm2 --min-slot-freq 10 --filename $conffile --batch-size 1 --epochs 50 --model $modelname --ptrmodel
  python -u src/generative_approach/training.py $conffile | tee $conffile.log
fi
