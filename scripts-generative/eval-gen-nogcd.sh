cd "$(dirname "$0")"
cd ..
source venv/bin/activate
export PYTHONPATH=./src/
if [ -n "$2" ]; then
    export configdate=$2
else
    export configdate=$(date '+%Y-%m-%d')
fi
ls -d1 cd /path/to/project/root/results_"$configdate"/gen/$1/*/ | xargs -n 1 -P 16 nice python src/full_eval.py --nodecoding --gen
