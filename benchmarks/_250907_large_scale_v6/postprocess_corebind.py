"""
Specify a particular job folder, and postprocess the corebind log.
"""
import os
from pathlib import Path

root_folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_2509907_large_scale_v6/logs.v1-core-bind/20250907_105233.jobinteractive-701223.d2-cp16-n2-b0-t131072"
nsys_folder = os.path.join(root_folder, "nsys-reps")

def export_nsys_to_sqlite(nsys_folder):
    # for each file in nsys_folder, export to sqlite using nsys export
    pass

