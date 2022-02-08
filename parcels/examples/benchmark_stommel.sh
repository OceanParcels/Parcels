#!/bin/zsh
# NOTE: I think this should be compatible with bash as well.

# This script uses the example_stommel.py script and parses the benchmark/timing results.
# The results are a simple table of # particles vs time, which can be plotted with gnuplot.
# The first line says which column is which.
# All timings are in seconds.
# The UNIX program awk is necessary to run this script.

# Iterate over these particles. If it takes too long, then just cancel it whenever and
# the results should still be in the file
N_PART=("1" "10" "100" "1000" "10000" "100000" "1000000" "10000000")

if [[ $# -lt 1 ]]; then
    print "Need at least one argument: name of run"
    exit 192
fi

# Put the results in a file called bench/{name}.txt (I used 'jit', 'scipy' and 'numba').
mkdir -p "bench"
BENCH_FILE="bench/$1.txt"
rm -f $BENCH_FILE

# After parsing the name, the remaining arguments are passed through to the example_stommel.py script
shift


for P in ${N_PART[*]}; do
    FIRST_LINE="#"
    TIMINGS="$P"

    # Run the stommel script with the here (<<<) operator and read the results line by line.
    while IFS= read -r LINE; do
        # Split the results into an array.
        LINE_ARR=(`echo $LINE`)

        # If the result is in seconds, life is simple.
        if [[ ${LINE_ARR[-1]} == "s" ]]; then
            FIRST_LINE+=" ${LINE_ARR[-4]}"
            TIMINGS+=" ${LINE_ARR[-2]}"
        else
            FIRST_LINE+=" ${LINE_ARR[-3]}"
            TIME_STR=${LINE_ARR[-1]}
            # Convert the time string into seconds
            SECS=`echo $TIME_STR | awk -F: '{ print ($1 * 3600) + ($2*60) + $3 }'`
            TIMINGS+=" $SECS"
        fi
    done <<<`python example_stommel.py -p $P $@ | tail -7`

    # In the first iteration, add the description header to the file.
    if [[ "$P" == "${N_PART[1]}" ]]; then
        echo "$FIRST_LINE" >> $BENCH_FILE
    fi

    # Append the timings for the current particle count.
    echo "$TIMINGS" >> $BENCH_FILE
done
