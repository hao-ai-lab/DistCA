for f in logs.v4-compare/*/benchmark.raw.jsonl; do
    echo "==== $f ====="
    cat $f | cut -c1-50 | grep duration
    echo
done