nsys stats nsys-profile/d2.nsys-rep --report nvtxppsum
nsys stats nsys-profile/d2.nsys-rep --report cuda_gpu_kern_sum --format table --force-export=true

nsys export --type sqlite --output nsys-profile/d2.sqlite nsys-profile/d2.nsys-rep







