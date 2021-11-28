#! /bin/bash

for dir in `git status | grep "extracted_features/.*"`
do
    for filename in `ls $dir`
    do
        #echo "$dir/$filename"
        git add $dir/$filename
        git commit -m "update extracted features dir $dir file $filename"
        git push
    done
done
