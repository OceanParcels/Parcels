#!/bin/zsh

N_PART=("1" "10" "100" "1000" "10000" "100000" "1000000" "10000000")

ARGS=($@)

if [[ $# -lt 1 ]]; then
    print "Need at least one argument: name of run"
    exit 192
fi

mkdir -p "bench"
BENCH_FILE="bench/$1.txt"
rm -f $BENCH_FILE

TMP_FILE="bench_temp.txt"

shift


for P in ${N_PART[*]}; do
    FIRST_LINE="#"
    TIMINGS="$P"
    while IFS= read -r LINE; do
        LINE_ARR=(`echo $LINE`)
        if [[ ${LINE_ARR[-1]} == "s" ]]; then
            FIRST_LINE+=" ${LINE_ARR[-4]}"
            TIMINGS+=" ${LINE_ARR[-2]}"
        else
            FIRST_LINE+=" ${LINE_ARR[-3]}"
            TIME_STR=${LINE_ARR[-1]}
            SECS=`echo $TIME_STR | awk -F: '{ print ($1 * 3600) + ($2*60) + $3 }'`
            TIMINGS+=" $SECS"
        fi
    done <<<`python example_stommel.py -p $P $@ | tail -7`
    if [[ "$P" == "${N_PART[1]}" ]]; then
        echo "$FIRST_LINE" >> $BENCH_FILE
    fi
    echo "$TIMINGS" >> $BENCH_FILE
#    LINE=(` | tail -1`)
#    print ${P} ${LINE[-2]} >> $BENCH_FILE
done
