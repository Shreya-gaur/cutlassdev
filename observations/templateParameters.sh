#!bin/bash

rm -f testingcommand.txt

exec 3>&1 1>>testingcommand.txt 2>&1

LINE=1

while read -r CURRENT_LINE
	do
		echo "Parameter is $CURRENT_LINE"
		sed -i "/DebugValue</c\		DebugValue< ${CURRENT_LINE} >::kStrided;" ../include/cutlass/conv/threadblock/*eqanalytic.h
		(cd ../build/examples/09_turing_tensorop_conv2dfprop && make) >> testingcommand.txt
		((LINE++))

done < templateParameters.txt


LINE=1

rm -f output.txt

while read -r CURRENT_LINE
	do
		echo " ------ ${CURRENT_LINE} ----- " >> output.txt

		grep "DebugValue<" testingcommand.txt | sed -n ${LINE}p >> output.txt

		echo " -------------------------------------------------------  " >> output.txt

		echo " " >> output.txt

		((LINE++))	
done < templateParameters.txt
