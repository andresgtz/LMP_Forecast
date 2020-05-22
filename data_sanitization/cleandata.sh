#!/bin/sh
# 
# Author: Andres Gutierrez
#
# Cleans original data to select only the nodes listed in the gawk command.
# Execution should be performed in a original data copy directory, as transformations are in place.
#
#

for file in *.csv; do
	echo $file
	gawk -i inplace -F',' '{ if ((FNR==1) || ($3 == "06LDC-115") || ($3 == "06ETK-115") || ($3 == "03ADR-69") || ($3 == "03GYS-115") || ($3 == "01INU-115") ||  ($3 == "01PIR-85") || ($3 == "08PLY-115") || ($3 == "08CHS-34.5") || ($3 == "08CHB-34.5") || ($3 == "08COZ-34.5") || ($3 == "04EFU-115") || ($3 == "06PUO-115")) print $0;}' $file
done
