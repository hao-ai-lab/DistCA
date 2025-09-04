
root_folder="logs.v4"
for folder in $root_folder/2025*; do
    log_folder="$folder/logs"
    a=$(grep -m 1 'bootstrap_net_recv' $log_folder/fs-mbz-gpu-*.out)
    if [ -n "$a" ]; then
        echo "--------------------------------"
        echo $folder
	grep 'bootstrap_net_recv' $log_folder/fs-mbz-gpu-*.out \
		  | sed -E 's#.*/(fs-mbz-gpu-[0-9]+)\..*#\1#' \
		    | sort -u
        echo "--------------------------------"
        echo
    fi

done
