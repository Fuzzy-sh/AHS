#!/bin/bash


echo Starting inclusion exclusion and cohort creation from top level Fuzzy script
# print out the date and time for now. 
date

echo ------------------------------------------------------------------------
# ########################################################################################
# Submit the job array
jobID1=$(sbatch  exclusion_cohortCreation.sh)

echo $jobID1

