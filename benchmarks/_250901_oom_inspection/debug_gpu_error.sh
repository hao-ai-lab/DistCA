

log_folder="/mnt/weka/home/yonghao.zhuang/jd/d2/tests/logs/20250903_223051"
grep -m 1 'cudaHostRegister' $log_folder/fs-mbz-gpu-*.out
if [ -n "$a" ]; then
    echo "--------------------------------"
    echo $folder
    grep 'bootstrap_net_recv' $log_folder/fs-mbz-gpu-*.out | sort | uniq
    echo "--------------------------------"
    echo
fi
