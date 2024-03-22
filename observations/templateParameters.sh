#!bin/bash

rm -f testingcommand.txt

exec 3>&1 1>>testingcommand.txt 2>&1

LINE=1

while read -r CURRENT_LINE
	do
		echo "Parameter is $CURRENT_LINE"
		sed -i "/DebugValue</c\		DebugValue< ${CURRENT_LINE} >::kStrided;" ../include/cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_eqanalytic.h
		(cd ../build/examples/09_turing_tensorop_conv2dfprop && make) >> testingcommand.txt
		((LINE++))

done < templateParameters.txt


LINE=1

rm -f output.txt

while read -r CURRENT_LINE
	do
		echo " ------ ${CURRENT_LINE} ----- " >> output_fil.txt

		grep "DebugValue<" testingcommand.txt | sed -n ${LINE}p >> output_fil.txt

		echo " -------------------------------------------------------  " >> output_fil.txt

		echo " " >> output_fil.txt

		((LINE++))	
done < templateParameters.txt
