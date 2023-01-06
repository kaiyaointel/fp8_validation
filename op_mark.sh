while read line
do
    mark=-1
    if [[ ${line} =~ "LayerNorm" ]]; then
        mark=4
    elif [[ ${line} =~ "mbedding" ]]; then
        mark=3
    elif [[ ${line} =~ "atmul" ]]; then
        mark=2
    elif [[ ${line} =~ "Linear" ]]; then
        mark=1
    elif [[ ${line} =~ "Conv" ]]; then
        mark=1
    fi
    echo ${line} ${mark}
done < a.log
