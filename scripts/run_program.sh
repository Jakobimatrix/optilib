#!/bin/bash
exe="../build/executable/optimization_samples"
if [ -e $exe ]; then
    exec $exe
else
    echo "Cannot find executable $exe."
fi

