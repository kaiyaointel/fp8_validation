set -x

rm -rf logs
mkdir logs

model_all="\
    bloomz-7b1
"
model_list=($(echo "${model_all}" |sed 's/,/ /g'))

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

setting_all="\
    fp8_static_e5m2,\
    fp8_static_e4m3,\
    fp8_static_e3m4,\
    fp8_dynamic_e4m3,\
    fp8_dynamic_e3m4,\
"
setting_list=($(echo "${setting_all}" |sed 's/,/ /g'))

dataset_all="\
    rte_super_glue,
"
dataset_list=($(echo "${dataset_all}" |sed 's/,/ /g'))

case_all="1,2,3,4,5"
case_list=($(echo "${case_all}" |sed 's/,/ /g'))

for model in ${model_list[@]}
do
    model_log=${model}
    if [[ ${model} =~ "/" ]]; then
        model_log=${model//'/'/'-'}
    fi

    for task in ${dataset_list[@]}
    do
        for setting in ${setting_list[@]}
        do
            if [ ${setting} == "int8_static" ]; then
                feature="pytorch_inc_static_quant_fx"
                fp8_data_format="e5m2"
            elif [ ${setting} == "int8_dynamic" ]; then
                feature="pytorch_inc_dynamic_quant"
                fp8_data_format="e5m2"
            elif [ ${setting} == "fp8_static_e5m2" ]; then
                feature="pytorch_inc_static_quant_fx_fp8"
                fp8_data_format="e5m2"
            elif [ ${setting} == "fp8_static_e4m3" ]; then
                feature="pytorch_inc_static_quant_fx_fp8"
                fp8_data_format="e4m3"
            elif [ ${setting} == "fp8_static_e3m4" ]; then
                feature="pytorch_inc_static_quant_fx_fp8"
                fp8_data_format="e3m4"
            elif [ ${setting} == "fp8_dynamic_e5m2" ]; then
                feature="pytorch_inc_dynamic_quant_fp8"
                fp8_data_format="e5m2"
            elif [ ${setting} == "fp8_dynamic_e4m3" ]; then
                feature="pytorch_inc_dynamic_quant_fp8"
                fp8_data_format="e4m3"
            elif [ ${setting} == "fp8_dynamic_e3m4" ]; then
                feature="pytorch_inc_dynamic_quant_fp8"
                fp8_data_format="e3m4"
            fi

            for case in ${case_list[@]}
            do
                if [ ${case} == "1" ]; then
                    export FP8_OP_TYPE_LIST="['linear', 'conv2d']"
                elif [ ${case} == "2" ]; then
                    export FP8_OP_TYPE_LIST="['linear', 'conv2d', 'bmm', 'amm', 'mm']"
                elif [ ${case} == "3" ]; then
                    export FP8_OP_TYPE_LIST="['linear', 'conv2d', 'bmm', 'amm', 'mm', 'embedding', 'embeddingbag']"
                elif [ ${case} == "4" ]; then
                    export FP8_OP_TYPE_LIST="['linear', 'conv2d', 'bmm', 'amm', 'mm', 'embedding', 'embeddingbag','layernorm']"
                elif [ ${case} == "5" ]; then
                    export FP8_OP_TYPE_LIST="['linear', 'conv2d', 'bmm', 'amm', 'mm', 'embedding', 'embeddingbag','layernorm']"
                    export DISABLE_FIRST_CONV=True
                    export DISABLE_LAST_LINEAR=True
                fi
            
                log_path="./logs/${model_log}-${task}-${setting}-${case}.log"
                
                if [ ${task} == "rte_super_glue" ]; then
                    task="rte"
                    task_hub="super_glue"
                elif [ ${task} == "cb_super_glue" ]; then
                    task="cb"
                    task_hub="super_glue"
                elif [ ${task} == "rte_glue" ]; then
                    task="rte"
                    task_hub="glue"
                elif [ ${task} == "qnli_glue" ]; then
                    task="qnli"
                    task_hub="glue"
                fi
                
                python -c 'from neural_coder import enable; result, _, _ = enable(code="https://github.com/kaiyaointel/transformers/blob/patch-1/examples/pytorch/text-classification/run_superglue.py", args="--model_name_or_path '${model}' --task_name '${task}' --task_hub_name '${task_hub}' --do_eval --output_dir result", features=["'${feature}'"], fp8_data_format="'${fp8_data_format}'", run_bench=True, use_inc=True,); print("acc_delta:", result[5]); print("acc_fp32:", result[6]); print("acc_int8:", result[7]);'\
                2>&1 | tee ${log_path}

                acc_delta=$(grep "acc_delta:" ${log_path} | sed -e 's/.*acc_delta//;s/[^-0-9.]//g')
                acc_fp32=$(grep "acc_fp32:" ${log_path} | sed -e 's/.*acc_fp32//;s/[^0-9.]//g')
                acc_int8=$(grep "acc_int8:" ${log_path} | sed -e 's/.*acc_int8//;s/[^0-9.]//g')

                echo ${model} ${task} ${setting} ${case} ${acc_delta} ${acc_fp32} ${acc_int8} | tee -a ./logs/summary.log
            done
        done
    done
done

cat ./logs/summary.log
