#! /bin/bash


for foldername in `ls .`;
do
    if [ $foldername = 'push.sh' ];
    then
        continue
    fi
    echo $foldername
    for filename in `ls $foldername`;
    do
        echo "$foldername/$filename"
        git add $foldername/$filename
        git commit -m "add extracted features $foldername/$filename"
        git push
    done
done