#!/bin/sh

for file in *.csv; do
	echo $file
	gawk -i inplace -F',' '{ if ((FNR==1) || ($3 == "08COZ-34.5") || ($3 == "04EFU-115") || ($3 == "06PUO-115")) print $0;}' $file
done


