#! /bin/bash

for dir in `git status | grep "extracted_features/.*"`
do
    if [ $dir = "modified:" ]; then
	continue
    fi
    echo "$dir"
    for filename in `ls $dir`
    do
        #echo "$dir/$filename"
	#echo "$filename"
        git add $filename
        git commit -m "update extracted features dir $dir file $filename"
        git push
	#exit
    done
done
