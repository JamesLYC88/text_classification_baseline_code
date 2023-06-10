format_list=(linear nn hier)

format=$1

if [ $# -eq 0 ]; then
    for format in "${format_list[@]}"; do
        python3 generate_data.py -f ${format}
    done
else
    if [[ ! " ${format_list[*]} " =~ " ${format} " ]]; then
        echo "Invalid argument! Format ${format} is not in (${format_list[*]})."
        exit
    else
        python3 generate_data.py -f ${format}
    fi
fi
