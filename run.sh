set -x

mkdir logs

model_all="\
    alexnet,\
    densenet121,densenet161,densenet169,densenet201,\
    efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,efficientnet_b7,\
    fbnetc_100,\
    googlenet,\
    inception_v3,\
    mnasnet0_5,mnasnet0_75,mnasnet1_0,mnasnet1_3,\
    mobilenet_v2,mobilenet_v3_small,mobilenet_v3_large,\
    resnet18,resnet34,resnet50,resnet101,resnet152,\
    resnext50_32x4d,resnext101_32x8d,\
    shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5,shufflenet_v2_x2_0,\
    spnasnet_100,\
    squeezenet1_0,squeezenet1_1,\
    vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,\
    wide_resnet50_2,wide_resnet101_2,\
    vit_b_16,vit_b_32,vit_l_16,vit_l_32,\
    convnext_tiny,convnext_small,convnext_base,convnext_large,\
    regnet_y_400mf,regnet_y_800mf,regnet_y_1_6gf,regnet_y_3_2gf,regnet_y_8gf,regnet_y_16gf,regnet_y_32gf,regnet_x_400mf,regnet_x_800mf,regnet_x_1_6gf,regnet_x_3_2gf,regnet_x_8gf,regnet_x_16gf,regnet_x_32gf,\
"

model_list=($(echo "${model_all}" |sed 's/,/ /g'))

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

setting_all="\
    int8_static,\
    int8_dynamic,\
    fp8_static_e5m2,\
    fp8_static_e4m3,\
    fp8_dynamic_e5m2,\
    fp8_dynamic_e4m3,\
"
setting_list=($(echo "${setting_all}" |sed 's/,/ /g'))

for model in ${model_list[@]}
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
        elif [ ${setting} == "fp8_dynamic_e5m2" ]; then
            feature="pytorch_inc_dynamic_quant_fp8"
            fp8_data_format="e5m2"
        elif [ ${setting} == "fp8_dynamic_e4m3" ]; then
            feature="pytorch_inc_dynamic_quant_fp8"
            fp8_data_format="e4m3"
        fi

        log_path="./logs/${model}-${setting}.log"

        python -c '\
            from neural_coder import enable
            result, _, _ = enable(
                code="https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py",
                args="--model_name_or_path albert-base-v2 --task_name sst2 --do_eval --output_dir result",
                features=[${feature}],
                fp8_data_format=${fp8_data_format},
                run_bench=True,
                use_inc=True,
            )
            print("acc_delta:", result[5])
            print("acc_fp32:", result[6])
            print("acc_int8:", result[7])
        '\
        2>&1 | tee ${log_path}
    
        acc_delta=$(grep "acc_delta:" ${log_path} | sed -e 's/.*acc_delta//;s/[^0-9.]//g')
        acc_fp32=$(grep "acc_fp32:" ${log_path} | sed -e 's/.*acc_fp32//;s/[^0-9.]//g')
        acc_int8=$(grep "acc_int8:" ${log_path} | sed -e 's/.*acc_int8//;s/[^0-9.]//g')
        
        echo ${model} ${setting} ${acc_delta} ${acc_fp32} ${acc_int8} | tee -a ./logs/summary.log
    done
done

cat ./logs/summary.log
