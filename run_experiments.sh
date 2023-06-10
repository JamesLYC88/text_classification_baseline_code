data_list=(ecthr_a ecthr_b scotus eurlex ledgar unfair_tos)
algo_list=(1vsrest thresholding cost_sensitive bert_default bert_tuned bert_reproduced)

data=$1
algo=$2

if [[ ! " ${data_list[*]} " =~ " ${data} " ]]; then
    echo "Invalid argument! Data ${data} is not in (${data_list[*]})."
    exit
fi

if [[ ! " ${algo_list[*]} " =~ " ${algo} " ]]; then
    echo "Invalid argument! Algorithm ${algo} is not in (${algo_list[*]})."
    exit
fi

linear_algo_list=(1vsrest thresholding cost_sensitive)
bert_algo_list=(bert_default bert_tuned bert_reproduced)

multilabel_unlabeled_data_list=(ecthr_a ecthr_b unfair_tos)
multilabel_labeled_data_list=(eurlex)
multiclass_labeled_data_list=(scotus ledgar)

if [[ " ${linear_algo_list[*]} " =~ " ${algo} " ]]; then
    if [[ " ${multilabel_unlabeled_data_list[*]} " =~ " ${data} " ]]; then
        python3 main.py --config config/${data}/l2svm.yml --linear_technique ${algo} --zero
    elif [[ " ${multilabel_labeled_data_list[*]} " =~ " ${data} " ]]; then
        python3 main.py --config config/${data}/l2svm.yml --linear_technique ${algo}
    elif [[ " ${multiclass_labeled_data_list[*]} " =~ " ${data} " ]]; then
        python3 main.py --config config/${data}/l2svm.yml --linear_technique ${algo} --multi_class
    else
        echo "Should never reach here..."
        exit
    fi
elif [[ " ${bert_algo_list[*]} " =~ " ${algo} " ]]; then
    if [[ " ${multilabel_unlabeled_data_list[*]} " =~ " ${data} " ]]; then
        python3 main.py --config config/${data}/${algo}.yml --zero --seed 1
    elif [[ " ${multilabel_labeled_data_list[*]} " =~ " ${data} " ]]; then
        python3 main.py --config config/${data}/${algo}.yml --seed 1
    elif [[ " ${multiclass_labeled_data_list[*]} " =~ " ${data} " ]]; then
        huggingface_trainer_algo_list=(bert_reproduced)
        if [[ ! " ${huggingface_trainer_algo_list[*]} " =~ " ${algo} " ]]; then
            python3 main.py --config config/${data}/${algo}.yml --multi_class --enable_ce_loss --seed 1
        else
            python3 main.py --config config/${data}/${algo}.yml --multi_class --enable_ce_loss --seed 1 --enable_transformer_trainer
        fi
    else
        echo "Should never reach here..."
        exit
    fi
fi
