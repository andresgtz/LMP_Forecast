#!/bin/sh

for file in *.csv; do
	echo $file
	gawk -i inplace -F',' '{ if ((FNR==1) || ($3 == "08CHS-34.5") || ($3 == "03STG-115") || ($3 == "04PLD-230")) print $0;}' $file
done


