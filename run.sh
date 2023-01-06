set -x

rm -rf logs
mkdir logs

pool=${1} # vision or hf

if [ ${pool} == "vision" ]; then
    model_all="\
        resnet18,alexnet,\
        densenet121,densenet161,\
        efficientnet_b0,efficientnet_b1,efficientnet_b2,\
        googlenet,\
        mnasnet0_5,mnasnet0_75,\
        mobilenet_v2,mobilenet_v3_small,\
        resnet34,resnet50,resnet101,\
        resnext50_32x4d,resnext101_32x8d,\
        shufflenet_v2_x0_5,shufflenet_v2_x1_0,\
        spnasnet_100,\
        squeezenet1_0,squeezenet1_1,\
        vgg11,vgg11_bn,vgg13,vgg16,vgg19,\
        wide_resnet50_2,wide_resnet101_2,\
        vit_b_16,vit_b_32,vit_l_16,vit_l_32,\
        convnext_tiny,convnext_small,convnext_base,convnext_large,\
        regnet_y_400mf,regnet_y_800mf,regnet_y_1_6gf,regnet_x_400mf,regnet_x_800mf,regnet_x_1_6gf,\
    "
elif [ ${pool} == "hf" ]; then
    # finetuned
    model_all="\
        cointegrated/roberta-large-cola-krishna2020,\
        ModelTC/bert-base-uncased-cola,\
        textattack/roberta-base-CoLA,\
        textattack/distilbert-base-uncased-CoLA,\
        textattack/bert-base-uncased-CoLA,\
        naitian/bert-cola-finetune,\
        textattack/distilbert-base-cased-CoLA,\
        gchhablani/bert-base-cased-finetuned-cola,\
        09panesara/distilbert-base-uncased-finetuned-cola,\
        vicl/canine-c-finetuned-cola,\
        2umm3r/distilbert-base-uncased-finetuned-cola,\
        TehranNLP-org/bert-base-uncased-avg-cola-2e-5-42,\
        yevheniimaslov/deberta-v3-large-cola,\
        Alireza1044/mobilebert_cola,\
        howey/electra-base-cola,\
        kamivao/autonlp-cola_gram-208681,\
        yoshitomo-matsubara/bert-base-uncased-cola,\
        yoshitomo-matsubara/bert-base-uncased-cola_from_bert-large-uncased-cola,\
        WillHeld/bert-base-cased-cola,\
        JeremiahZ/bert-base-uncased-cola,\
        yoshitomo-matsubara/bert-large-uncased-cola,\
        mrm8488/deberta-v3-small-finetuned-cola,\
        gchhablani/bert-large-cased-finetuned-cola,\
        graphcore-rahult/roberta-base-finetuned-cola,\
        Modfiededition/bert-fine-tuned-cola,\
        textattack/albert-base-v2-CoLA,\
        Aktsvigun/electra-large-cola,\
        conanoutlook/distilbert-base-uncased-finetuned-cola,\
        EhsanAghazadeh/bert-large-uncased-CoLA_A,\
        VirenS13117/distilbert-base-uncased-finetuned-colaspPeach/nli-distilroberta-base-finetuned-cola,\
        Rocketknight1/distilbert-base-uncased-finetuned-cola,\
        choeunsoo/bert-base-uncased-finetuned-cola,\
        Jeevesh8/bert_ft_cola-83,\
        avb/bert-base-uncased-finetuned-cola,\
        Jeevesh8/bert_ft_cola-6,\
        Jeevesh8/bert_ft_cola-18,\
        Jeevesh8/bert_ft_cola-34,\
        Jeevesh8/bert_ft_cola-37,\
        Jeevesh8/bert_ft_cola-38,\
        Jeevesh8/bert_ft_cola-49,\
        Jeevesh8/bert_ft_cola-53,\
        Jeevesh8/bert_ft_cola-58,\
        Jeevesh8/bert_ft_cola-91,\
        Jeevesh8/bert_ft_cola-95,\
        Jeevesh8/bert_ft_cola-99,\
        Jeevesh8/6ep_bert_ft_cola-1,\
        Jeevesh8/6ep_bert_ft_cola-12,\
        Jeevesh8/6ep_bert_ft_cola-13,\
        Jeevesh8/6ep_bert_ft_cola-17,\
        Jeevesh8/6ep_bert_ft_cola-20,\
        textattack/bert-base-uncased-MRPC,\
        textattack/albert-base-v2-MRPC,\
        Intel/bert-base-uncased-mrpc,\
        Intel/electra-small-discriminator-mrpc,\
        M-FAC/bert-mini-finetuned-mrpc,\
        hf-internal-testing/mrpc-bert-base-cased,\
        textattack/roberta-base-MRPC,\
        Intel/roberta-base-mrpc,\
        Intel/xlnet-base-cased-mrpc,\
        Intel/roberta-base-mrpc-int8-static,\
        Intel/bert-base-uncased-mrpc-int8-qat,\
        sgugger/finetuned-bert-mrpc,\
        textattack/distilbert-base-uncased-MRPC,\
        yoshitomo-matsubara/bert-base-uncased-mrpc,\
        yoshitomo-matsubara/bert-large-uncased-mrpc,\
        Intel/MiniLM-L12-H384-uncased-mrpc-int8-static,\
        Intel/MiniLM-L12-H384-uncased-mrpc-int8-qat,\
        gchhablani/bert-base-cased-finetuned-mrpc,\
        gchhablani/bert-large-cased-finetuned-mrpc,\
        Intel/bert-base-uncased-mrpc-int8-static,\
        patrickvonplaten/bert-base-cased_fine_tuned_glue_mrpc_demo,\
        Intel/bert-base-uncased-mrpc-int8-dynamic,\
        yoshitomo-matsubara/bert-base-uncased-mrpc_from_bert-large-uncased-mrpc,\
        Intel/MiniLM-L12-H384-uncased-mrpc,\
        JeremiahZ/bert-base-uncased-mrpc,\
        howey/electra-base-mrpc,\
        textattack/distilbert-base-cased-MRPC,\
        sgugger/glue-mrpc,\
        sgugger/bert-finetuned-mrpc,\
        course5i/SEAD-L-6_H-256_A-8-mrpccourse5i/SEAD-L-6_H-384_A-12-mrpc,\
        ModelTC/bert-base-uncased-mrpc,\
        Intel/xlm-roberta-base-mrpc,\
        Jinchen/roberta-base-finetuned-mrpc,\
        Shularp/finetuned-bert-mrpc,\
        ljh1/mrpc,\
        Yanael/bert-finetuned-mrpc,\
        anuj55/distilbert-base-uncased-finetuned-mrpc,\
        vicl/distilbert-base-uncased-finetuned-mrpc,\
        ArafatBHossain/bert-base-mrpc,\
        Maelstrom77/roberta-large-mrpc,\
        mrm8488/data2vec-text-base-finetuned-mrpc,\
        izboy250/finetuned-bert-mrpc,\
        pinecone/bert-mrpc-cross-encoder,\
        Intel/camembert-base-mrpc,\
        Applemoon/bert-finetuned-mrpc,\
        oumeima/finetuned-bert-mrpc,\
        JeremiahZ/roberta-base-mrpc,\
        AdapterHub/bert-base-uncased-pf-mrpc,\
        Ruizhou/bert-base-uncased-finetuned-mrpc,\
        Intel/bart-large-mrpc,\
        philschmid/tiny-bert-sst2-distilled,\
        howey/roberta-large-sst2,\
        echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid,\
        valhalla/bart-large-sst2,\
        bhadresh-savani/distilbert-base-uncased-sentiment-sst2,\
        philschmid/MiniLM-L6-H384-uncased-sst2,\
        TehranNLP-org/bert-base-uncased-avg-sst2-2e-5-42,\
        winegarj/distilbert-base-uncased-finetuned-sst2,\
        assemblyai/distilbert-base-uncased-sst2,\
        yoshitomo-matsubara/bert-base-uncased-sst2,\
        course5i/SEAD-L-6_H-256_A-8-sst2,\
        Alireza1044/albert-base-v2-sst2,\
        gchhablani/bert-base-cased-finetuned-sst2,\
        gchhablani/fnet-large-finetuned-sst2,\
        moshew/bert-mini-sst2-distilled,\
        yoshitomo-matsubara/bert-large-uncased-sst2,\
        philschmid/roberta-large-sst2,\
        SetFit/deberta-v3-base__sst2__all-train,\
        howey/electra-base-sst2,\
        assemblyai/bert-large-uncased-sst2,\
        Intel/bert-mini-sst2-distilled-sparse-90-1X4-block,\
        M-FAC/bert-tiny-finetuned-sst2,\
        WillHeld/bert-base-cased-sst2,\
        yoshitomo-matsubara/bert-base-uncased-sst2_from_bert-large-uncased-sst2,\
        gchhablani/fnet-base-finetuned-sst2,\
        doyoungkim/bert-base-uncased-finetuned-sst2,\
        howey/bert-base-uncased-sst2,\
        JeremiahZ/bert-base-uncased-sst2,\
        TehranNLP-org/bert-base-uncased-cls-sst2,\
        TehranNLP-org/bert-large-sst2course5i/SEAD-L-6_H-384_A-12-sst2,\
        rcorkill/RoBERTRAMa-SST2,\
        mrm8488/deberta-v3-small-finetuned-sst2,\
        michelecafagna26/t5-base-finetuned-sst2-sentiment,\
        CaffreyR/sst2_lora_bert,\
        zeroshot/sst2-obert-sparse,\
        SetFit/deberta-v3-large__sst2__train-32-1,\
        Smith123/tiny-bert-sst2-distilled_L6_H128,\
        gokuls/bert-base-sst2,\
        SetFit/deberta-v3-large__sst2__train-16-0,\
        Bhumika/roberta-base-finetuned-sst2,\
        M-FAC/bert-mini-finetuned-sst2,\
        SetFit/MiniLM-L12-H384-uncased__sst2__all-train,\
        SetFit/deberta-v3-large__sst2__train-32-0,\
        EhsanAghazadeh/bert-based-uncased-sst2-e3,\
        Hieom/tiny-bert-sst2-distilled,\
        SetFit/distilbert-base-uncased__sst2__train-16-0,\
        SetFit/distilbert-base-uncased__sst2__train-16-1,\
        TehranNLP-org/electra-base-sst2,\
        zeroshot/sst2-distilbert-dense,\
        TehranNLP-org/xlnet-base-cased-avg-sst2-2e-5-63,\
        cross-encoder/stsb-distilroberta-base,\
        cross-encoder/stsb-roberta-large,\
        cross-encoder/stsb-roberta-base,\
        cross-encoder/stsb-TinyBERT-L-4,\
        yoshitomo-matsubara/bert-base-uncased-stsb,\
        yoshitomo-matsubara/bert-base-uncased-stsb_from_bert-large-uncased-stsb,\
        WillHeld/bert-base-cased-stsb,\
        gchhablani/bert-base-cased-finetuned-stsb,\
        yoshitomo-matsubara/bert-large-uncased-stsb,\
        efederici/cross-encoder-bert-base-stsb,\
        course5i/SEAD-L-6_H-384_A-12-stsb,\
        JeremiahZ/bert-base-uncased-stsb,\
        course5i/SEAD-L-6_H-256_A-8-stsb,\
        ModelTC/roberta-base-stsb,\
        Katsiaryna/stsb-distilroberta-base-finetuned_9th_auc_ce,\
        ModelTC/bart-base-stsb,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_151221-normal,\
        efederici/cross-encoder-umberto-stsb,\
        jkgrad/longformer-base-stsb,\
        WillHeld/roberta-base-stsb,\
        ModelTC/bert-base-uncased-stsb,\
        AdapterHub/bert-base-uncased-pf-stsb,\
        howey/electra-large-stsb,\
        howey/roberta-large-stsb,\
        ntrnghia/stsb_vn,\
        M-FAC/bert-mini-finetuned-stsb,\
        gchhablani/fnet-large-finetuned-stsb,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_151221-top3_op1,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_151221-top3_op3,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_161221-top3Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_40000-top3-BCE,\
        vicl/canine-s-finetuned-stsb,\
        vicl/distilbert-base-uncased-finetuned-stsb,\
        mrm8488/data2vec-text-base-finetuned-stsb,\
        JeremiahZ/roberta-base-stsb,\
        Sayan01/tiny-bert-stsb-distilled,\
        Team-PIXEL/pixel-base-finetuned-stsb,\
        Sayan01/stsb-distillbert,\
        Sayan01/stsb-distillbert-Direct,\
        DanNav/roberta-finetuned-stsb,\
        shivangi/STS-B_64_128_output,\
        AdapterHub/roberta-base-pf-stsb,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_151221-5-001,\
        Alireza1044/albert-base-v2-stsb,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_151221-top1,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_151221-top3,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_151221-top3_op2,\
        Katsiaryna/stsb-TinyBERT-L-4-finetuned_auc_40000-top3,\
        M-FAC/bert-tiny-finetuned-stsb,\
        gchhablani/fnet-base-finetuned-stsb,\
        textattack/bert-base-uncased-QNLI,\
        textattack/roberta-base-QNLI,\
        cross-encoder/qnli-electra-base,\
        cross-encoder/qnli-distilroberta-base,\
        yoshitomo-matsubara/bert-base-uncased-qnli,\
        gchhablani/bert-base-cased-finetuned-qnli,\
        WillHeld/bert-base-cased-qnli,\
        yoshitomo-matsubara/bert-base-uncased-qnli_from_bert-large-uncased-qnli,\
        yoshitomo-matsubara/bert-large-uncased-qnli,\
        JeremiahZ/bert-base-uncased-qnli,\
        course5i/SEAD-L-6_H-384_A-12-qnli,\
        course5i/SEAD-L-6_H-256_A-8-qnli,\
        textattack/distilbert-base-uncased-QNLI,\
        JeremiahZ/roberta-base-qnli,\
        Huffon/qnli,\
        Katsiaryna/qnli-electra-base-finetuned_9th_auc_ce,\
        Li/bert-base-uncased-qnli,\
        ModelTC/roberta-base-qnli,\
        princeton-nlp/CoFi-QNLI-s95,\
        ModelTC/bert-base-uncased-qnli,\
        anirudh21/albert-base-v2-finetuned-qnli,\
        gchhablani/fnet-base-finetuned-qnli,\
        Alireza1044/albert-base-v2-qnli,\
        ModelTC/bart-base-qnli,\
        Alireza1044/MobileBERT_Theseus-qnli,\
        princeton-nlp/CoFi-QNLI-s60,\
        howey/electra-base-qnli,\
        mrm8488/deberta-v3-small-finetuned-qnli,\
        Katsiaryna/qnli-electra-base-finetuned_auc,\
        savasy/bert-turkish-uncased-qnliKatsiaryna/qnli-electra-base-finetuned_auc,\
        M-FAC/bert-mini-finetuned-qnli,\
        Sayan01/tiny-bert-qnli-distilled,\
        Sayan01/qnli-distilled-bart-cross-roberta,\
        DanNav/roberta-finetuned-qnli,\
        Sayan01/tiny-bert-qnli128-distilled,\
        AdapterHub/bert-base-uncased-pf-qnli,\
        Alireza1044/mobilebert_QNLI,\
        WillHeld/roberta-base-qnli,\
        bioformers/bioformer-cased-v1.0-qnli,\
        textattack/xlnet-base-cased-QNLI,\
        negfir/distilbert-base-uncased-finetuned-qnli,\
        M-FAC/bert-tiny-finetuned-qnli,\
        howey/bert-base-uncased-qnli,\
        AdapterHub/roberta-base-pf-qnli,\
        howey/electra-large-qnli,\
        howey/roberta-large-qnli,\
        Team-PIXEL/pixel-base-finetuned-qnli,\
        mervenoyan/PubMedBERT-QNLI,\
        anirudh21/albert-large-v2-finetuned-qnli,\
        mrm8488/bert-uncased-finetuned-qnli,\
    "
elif [ ${pool} == "vit" ]; then
    # finetuned first
    model_all="\
        nateraw/vit-base-beans,\
        aaraki/vit-base-patch16-224-in21k-finetuned-cifar10,\
        farleyknight-org-username/vit-base-mnist,\
        akahana/vit-base-cats-vs-dogs,\
        nateraw/food,\
    "
fi

model_list=($(echo "${model_all}" |sed 's/,/ /g'))

export ftp_proxy="http://proxy-prc.intel.com:913"
export http_proxy="http://proxy-prc.intel.com:913"
export https_proxy="http://proxy-prc.intel.com:913"
export no_proxy=".intel.com"

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
    fp8_static_e3m4,\
    fp8_dynamic_e4m3,\
    fp8_dynamic_e3m4,\
"
setting_list=($(echo "${setting_all}" |sed 's/,/ /g'))

for model in ${model_list[@]}
do
    # clear cache for space
    rm -rf ~/.cache/torch/hub/checkpoints/
    rm -rf ~/.cache/huggingface

    if [ ${pool} == "hf" ]; then
        # hf finetune dataset
        model_t=$(echo ${model} | tr [A-Z] [a-z])
        if [[ ${model_t} =~ "cola" ]]; then
            task="cola"
        elif [[ ${model_t} =~ "sst2" ]]; then
            task="sst2"
        elif [[ ${model_t} =~ "mrpc" ]]; then
            task="mrpc"
        elif [[ ${model_t} =~ "stsb" ]]; then
            task="stsb"
        elif [[ ${model_t} =~ "qnli" ]]; then
            task="qnli"
        fi
    elif [ ${pool} == "vit" ]; then
        # vit finetune dataset
        if [ ${model} == "nateraw/vit-base-beans" ]; then
            task="beans"
        elif [ ${model} == "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10" ]; then
            task="cifar10"
        elif [ ${model} == "farleyknight-org-username/vit-base-mnist" ]; then
            task="mnist"
        elif [ ${model} == "akahana/vit-base-cats-vs-dogs" ]; then
            task="cats_vs_dogs"
        elif [ ${model} == "nateraw/food" ]; then
            task="nateraw/food101"
        fi
    fi

    model_log=${model}
    if [[ ${model} =~ "/" ]]; then
        model_log=${model//'/'/'-'}
    fi

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

        log_path="./logs/${model_log}-${setting}.log"

        if [ ${pool} == "vision" ]; then
            python -c 'from neural_coder import enable; result, _, _ = enable(code="https://github.com/pytorch/examples/blob/main/imagenet/main.py", args="-a '${model}' -e --pretrained /home2/pytorch-broad-models/imagenet/raw/", features=["'${feature}'"], fp8_data_format="'${fp8_data_format}'", run_bench=True, use_inc=True,); print("acc_delta:", result[5]); print("acc_fp32:", result[6]); print("acc_int8:", result[7]);'\
            2>&1 | tee ${log_path}
        elif [ ${pool} == "hf" ]; then
            python -c 'from neural_coder import enable; result, _, _ = enable(code="https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py", args="--model_name_or_path '${model}' --task_name '${task}' --do_eval --output_dir result", features=["'${feature}'"], fp8_data_format="'${fp8_data_format}'", run_bench=True, use_inc=True,); print("acc_delta:", result[5]); print("acc_fp32:", result[6]); print("acc_int8:", result[7]);'\
            2>&1 | tee ${log_path}
        elif [ ${pool} == "vit" ]; then
            python -c 'from neural_coder import enable; result, _, _ = enable(code="https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/image-classification/run_image_classification.py", args="--model_name_or_path '${model}' --dataset_name '${task}' --do_eval --output_dir result --remove_unused_columns False", features=["'${feature}'"], fp8_data_format="'${fp8_data_format}'", run_bench=True, use_inc=True,); print("acc_delta:", result[5]); print("acc_fp32:", result[6]); print("acc_int8:", result[7]);'\
            2>&1 | tee ${log_path}
        fi

        # log parsing
        acc_delta=$(grep "acc_delta:" ${log_path} | sed -e 's/.*acc_delta//;s/[^-0-9.]//g')
        acc_fp32=$(grep "acc_fp32:" ${log_path} | sed -e 's/.*acc_fp32//;s/[^0-9.]//g')
        acc_int8=$(grep "acc_int8:" ${log_path} | sed -e 's/.*acc_int8//;s/[^0-9.]//g')
        
		op_types="[]"
        op_types=$(grep "Suggested FP8 op types are" ${log_path} | sed -e 's/.*are://;s/; Acc.*//')

        find_str="Disable first conv and last linear"
        disable_first_last=0
        if [ `grep -c "${find_str}" ${log_path}` -ne '0' ];then
            disable_first_last=1
        fi

        echo ${model} ${setting} ${acc_delta} ${acc_fp32} ${acc_int8} ${op_types} ${disable_first_last} | tee -a ./logs/summary.log

    done

    # clear cache for space
    rm -rf ~/.cache/torch/hub/checkpoints/
    rm -rf ~/.cache/huggingface

done

cat ./logs/summary.log
