#!/bin/sh

for file in *.csv; do
	echo $file
	SED_ARGS="1,7d $file"
	sed -i $SED_ARGS
done