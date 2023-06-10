data_list=(ecthr_a ecthr_b scotus eurlex ledgar unfair_tos)

data=$1

if [[ ! " ${data_list[*]} " =~ " ${data} " ]]; then
    echo "Invalid argument! Data ${data} is not in (${data_list[*]})."
    exit
fi

python3 search_params.py --config config/${data}/bert_tune.yml
