#!/bin/bash

while IFS= read -r line; do
  X=${line%%_*}
  Y=${line##*_}
  #echo $X $Y
  sbatch restart_job.sbatch $X $Y
done < jobs_to_submit.txt

