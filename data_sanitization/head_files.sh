#!/bin/sh

for file in *.csv; do
	echo $file
	head -n3 $file
done