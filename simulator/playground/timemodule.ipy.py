# %%
from io import StringIO
import pandas as pd


file_content = """total_len	batch_size	tp	cp	latency
128	1	1	1	0.06
128	1	1	2	0.06
128	1	1	4	0.06
128	1	1	8	0.06
128	1	2	1	0.06
128	1	2	2	0.06
128	1	2	4	0.06
128	1	2	8	0.06
128	1	4	1	0.06
128	1	4	2	0.06
128	1	4	4	0.06
128	1	4	8	0.06
128	1	8	1	0.06
128	1	8	2	0.06
128	1	8	4	0.07
128	1	8	8	0.06
256	1	1	1	0.06
256	1	1	2	0.06
256	1	1	4	0.07
256	1	1	8	0.06
256	1	2	1	0.06
256	1	2	2	0.06
256	1	2	4	0.07
256	1	2	8	0.06
256	1	4	1	0.06
256	1	4	2	0.06
256	1	4	4	0.06
256	1	4	8	0.06
256	1	8	1	0.06
256	1	8	2	0.06
256	1	8	4	0.06
256	1	8	8	0.06
512	1	1	1	0.08
512	1	1	2	0.06
512	1	1	4	0.06
512	1	1	8	0.06
512	1	2	1	0.07
512	1	2	2	0.07
512	1	2	4	0.06
512	1	2	8	0.06
512	1	4	1	0.07
512	1	4	2	0.07
512	1	4	4	0.06
512	1	4	8	0.06
512	1	8	1	0.07
512	1	8	2	0.06
512	1	8	4	0.06
512	1	8	8	0.07
1024	1	1	1	0.16
1024	1	1	2	0.09
1024	1	1	4	0.07
1024	1	1	8	0.07
1024	1	2	1	0.1
1024	1	2	2	0.08
1024	1	2	4	0.07
1024	1	2	8	0.07
1024	1	4	1	0.08
1024	1	4	2	0.08
1024	1	4	4	0.07
1024	1	4	8	0.07
1024	1	8	1	0.08
1024	1	8	2	0.07
1024	1	8	4	0.07
1024	1	8	8	0.07
2048	1	1	1	0.31
2048	1	1	2	0.19
2048	1	1	4	0.11
2048	1	1	8	0.08
2048	1	2	1	0.21
2048	1	2	2	0.12
2048	1	2	4	0.09
2048	1	2	8	0.08
2048	1	4	1	0.15
2048	1	4	2	0.1
2048	1	4	4	0.09
2048	1	4	8	0.08
2048	1	8	1	0.11
2048	1	8	2	0.09
2048	1	8	4	0.09
2048	1	8	8	0.08
4096	1	1	1	0.92
4096	1	1	2	0.51
4096	1	1	4	0.28
4096	1	1	8	0.16
4096	1	2	1	0.53
4096	1	2	2	0.31
4096	1	2	4	0.17
4096	1	2	8	0.11
4096	1	4	1	0.35
4096	1	4	2	0.19
4096	1	4	4	0.12
4096	1	4	8	0.11
4096	1	8	1	0.24
4096	1	8	2	0.14
4096	1	8	4	0.12
4096	1	8	8	0.12
8192	1	1	1	3.13
8192	1	1	2	1.63
8192	1	1	4	0.88
8192	1	1	8	0.47
8192	1	2	1	1.71
8192	1	2	2	0.93
8192	1	2	4	0.51
8192	1	2	8	0.27
8192	1	4	1	0.98
8192	1	4	2	0.56
8192	1	4	4	0.29
8192	1	4	8	0.18
8192	1	8	1	0.63
8192	1	8	2	0.34
8192	1	8	4	0.2
8192	1	8	8	0.18
16384	1	1	1	11.85
16384	1	1	2	5.96
16384	1	1	4	3.05
16384	1	1	8	1.59
16384	1	2	1	6.06
16384	1	2	2	3.16
16384	1	2	4	1.69
16384	1	2	8	0.86
16384	1	4	1	3.26
16384	1	4	2	1.76
16384	1	4	4	0.93
16384	1	4	8	0.48
16384	1	8	1	1.87
16384	1	8	2	1.05
16384	1	8	4	0.52
16384	1	8	8	0.31
32768	1	1	1	46.51
32768	1	1	2	23.49
32768	1	1	4	11.83
32768	1	1	8	5.93
32768	1	2	1	23.59
32768	1	2	2	11.82
32768	1	2	4	6.03
32768	1	2	8	3.09
32768	1	4	1	12.13
32768	1	4	2	6.17
32768	1	4	4	3.21
32768	1	4	8	1.65
32768	1	8	1	6.37
32768	1	8	2	3.42
32768	1	8	4	1.82
32768	1	8	8	0.89
65536	1	1	1	195.51
65536	1	1	2	97.41
65536	1	1	4	47.17
65536	1	1	8	23.56
65536	1	2	1	98.18
65536	1	2	2	47.03
65536	1	2	4	23.83
65536	1	2	8	11.9
65536	1	4	1	47.74
65536	1	4	2	23.9
65536	1	4	4	12.22
65536	1	4	8	6.09
65536	1	8	1	24.21
65536	1	8	2	12.33
65536	1	8	4	6.39
65536	1	8	8	3.23
49152	1	1	1	111.39
49152	1	1	2	54.38
49152	1	1	4	27.39
49152	1	1	8	13.68
49152	1	2	1	55.79
49152	1	2	2	27.37
49152	1	2	4	14.15
49152	1	2	8	6.96
49152	1	4	1	27.47
49152	1	4	2	13.9
49152	1	4	4	7.03
49152	1	4	8	3.52
49152	1	8	1	14.06
49152	1	8	2	7.23
49152	1	8	4	3.66
49152	1	8	8	2.26
98304	1	1	1	463.5
98304	1	1	2	227.86
98304	1	1	4	114.07
98304	1	1	8	55.54
98304	1	2	1	232.67
98304	1	2	2	115.53
98304	1	2	4	56.88
98304	1	2	8	27.88
98304	1	4	1	116.77
98304	1	4	2	56.39
98304	1	4	4	28.02
98304	1	4	8	14.38
98304	1	8	1	58.44
98304	1	8	2	28.28
98304	1	8	4	14.51
98304	1	8	8	6.94
131072	1	1	1	842.39
131072	1	1	2	417.1
131072	1	1	4	207.66
131072	1	1	8	103.4
131072	1	2	1	422.24
131072	1	2	2	208.96
131072	1	2	4	104.16
131072	1	2	8	50.74
131072	1	4	1	212.28
131072	1	4	2	104.8
131072	1	4	4	51.52
131072	1	4	8	25.25
131072	1	8	1	107.35
131072	1	8	2	51.82
131072	1	8	4	25.97
131072	1	8	8	12.99"""

df = pd.read_csv(StringIO(file_content), sep="\t")

# %%
df.head()

# %%
df.tail()

# %%
df.columns

# %%
# Create a nested dictionary with (tp, cp) as the first key and total_len as the second key
from rich import print
time_dict = {}
for _, row in df.iterrows():
    key = (row['tp'], row['cp'])
    key = (int(key[0]), int(key[1]))
    if key not in time_dict:
        time_dict[key] = {}
    time_dict[key][int(row['total_len'])] = row['latency'].item()

for key, value in time_dict.items():
    time_dict[key] = {k: v for k, v in sorted(value.items())}


# Print the resulting dictionary
print(time_dict)

# %%



network_content = """gpu_type	num_gpu	op	nelem	dtype	latency(ms)
H100	2	allreduce	32	fp16	0.04
H100	2	allreduce	64	fp16	0.04
H100	2	allreduce	128	fp16	0.04
H100	2	allreduce	256	fp16	0.04
H100	2	allreduce	512	fp16	0.04
H100	2	allreduce	1024	fp16	0.04
H100	2	allreduce	2048	fp16	0.04
H100	2	allreduce	4096	fp16	0.04
H100	2	allreduce	8192	fp16	0.04
H100	2	allreduce	16384	fp16	0.04
H100	2	allreduce	32768	fp16	0.04
H100	2	allreduce	65536	fp16	0.04
H100	2	allreduce	131072	fp16	0.04
H100	2	allreduce	262144	fp16	0.04
H100	2	allreduce	524288	fp16	0.05
H100	2	allreduce	1048576	fp16	0.06
H100	2	allreduce	2097152	fp16	0.06
H100	2	allreduce	4194304	fp16	0.07
H100	2	allreduce	8388608	fp16	0.10
H100	2	allreduce	16777216	fp16	0.17
H100	2	allreduce	33554432	fp16	0.28
H100	2	allreduce	67108864	fp16	0.49
H100	2	allreduce	134217728	fp16	0.91
H100	2	allreduce	268435456	fp16	1.71
H100	2	allreduce	536870912	fp16	3.27
H100	2	allreduce	1073741824	fp16	6.32
H100	2	allreduce	2147483648	fp16	12.43
H100	2	allreduce	4294967296	fp16	24.94
H100	2	allreduce	8589934592	fp16	49.21
H100	2	allreduce	17179869184	fp16	98.21
H100	4	allreduce	32	fp16	0.08
H100	4	allreduce	64	fp16	0.08
H100	4	allreduce	128	fp16	0.08
H100	4	allreduce	256	fp16	0.08
H100	4	allreduce	512	fp16	0.08
H100	4	allreduce	1024	fp16	0.08
H100	4	allreduce	2048	fp16	0.08
H100	4	allreduce	4096	fp16	0.08
H100	4	allreduce	8192	fp16	0.08
H100	4	allreduce	16384	fp16	0.08
H100	4	allreduce	32768	fp16	0.08
H100	4	allreduce	65536	fp16	0.08
H100	4	allreduce	131072	fp16	0.09
H100	4	allreduce	262144	fp16	0.10
H100	4	allreduce	524288	fp16	0.08
H100	4	allreduce	1048576	fp16	0.16
H100	4	allreduce	2097152	fp16	0.21
H100	4	allreduce	4194304	fp16	0.10
H100	4	allreduce	8388608	fp16	0.20
H100	4	allreduce	16777216	fp16	0.21
H100	4	allreduce	33554432	fp16	0.35
H100	4	allreduce	67108864	fp16	0.78
H100	4	allreduce	134217728	fp16	1.30
H100	4	allreduce	268435456	fp16	2.33
H100	4	allreduce	536870912	fp16	4.54
H100	4	allreduce	1073741824	fp16	8.94
H100	4	allreduce	2147483648	fp16	17.77
H100	4	allreduce	4294967296	fp16	35.35
H100	4	allreduce	8589934592	fp16	70.59
H100	4	allreduce	17179869184	fp16	140.53
H100	8	allreduce	32	fp16	0.22
H100	8	allreduce	64	fp16	0.23
H100	8	allreduce	128	fp16	0.25
H100	8	allreduce	256	fp16	0.20
H100	8	allreduce	512	fp16	0.16
H100	8	allreduce	1024	fp16	0.11
H100	8	allreduce	2048	fp16	0.08
H100	8	allreduce	4096	fp16	0.05
H100	8	allreduce	8192	fp16	0.05
H100	8	allreduce	16384	fp16	0.05
H100	8	allreduce	32768	fp16	0.05
H100	8	allreduce	65536	fp16	0.05
H100	8	allreduce	131072	fp16	0.05
H100	8	allreduce	262144	fp16	0.71
H100	8	allreduce	524288	fp16	0.54
H100	8	allreduce	1048576	fp16	0.58
H100	8	allreduce	2097152	fp16	0.67
H100	8	allreduce	4194304	fp16	0.63
H100	8	allreduce	8388608	fp16	0.63
H100	8	allreduce	16777216	fp16	0.70
H100	8	allreduce	33554432	fp16	0.71
H100	8	allreduce	67108864	fp16	0.69
H100	8	allreduce	134217728	fp16	1.14
H100	8	allreduce	268435456	fp16	2.13
H100	8	allreduce	536870912	fp16	4.09
H100	8	allreduce	1073741824	fp16	8.07
H100	8	allreduce	2147483648	fp16	15.83
H100	8	allreduce	4294967296	fp16	31.45
H100	8	allreduce	8589934592	fp16	62.93
H100	8	allreduce	17179869184	fp16	125.28
H100	2	allgather	32	fp16	0.12
H100	2	allgather	64	fp16	0.14
H100	2	allgather	128	fp16	0.11
H100	2	allgather	256	fp16	0.13
H100	2	allgather	512	fp16	0.12
H100	2	allgather	1024	fp16	0.10
H100	2	allgather	2048	fp16	0.10
H100	2	allgather	4096	fp16	0.10
H100	2	allgather	8192	fp16	0.09
H100	2	allgather	16384	fp16	0.10
H100	2	allgather	32768	fp16	0.10
H100	2	allgather	65536	fp16	0.09
H100	2	allgather	131072	fp16	0.25
H100	2	allgather	262144	fp16	0.27
H100	2	allgather	524288	fp16	0.25
H100	2	allgather	1048576	fp16	0.19
H100	2	allgather	2097152	fp16	0.25
H100	2	allgather	4194304	fp16	0.33
H100	2	allgather	8388608	fp16	0.37
H100	2	allgather	16777216	fp16	0.38
H100	2	allgather	33554432	fp16	0.64
H100	2	allgather	67108864	fp16	0.87
H100	2	allgather	134217728	fp16	1.53
H100	2	allgather	268435456	fp16	2.84
H100	2	allgather	536870912	fp16	5.38
H100	2	allgather	1073741824	fp16	10.37
H100	2	allgather	2147483648	fp16	45.10
H100	2	allgather	4294967296	fp16	145.34
H100	2	allgather	8589934592	fp16	314.83
H100	4	allgather	32	fp16	0.12
H100	4	allgather	64	fp16	0.11
H100	4	allgather	128	fp16	0.11
H100	4	allgather	256	fp16	0.10
H100	4	allgather	512	fp16	0.10
H100	4	allgather	1024	fp16	0.10
H100	4	allgather	2048	fp16	0.10
H100	4	allgather	4096	fp16	0.12
H100	4	allgather	8192	fp16	0.11
H100	4	allgather	16384	fp16	0.11
H100	4	allgather	32768	fp16	0.12
H100	4	allgather	65536	fp16	0.41
H100	4	allgather	131072	fp16	0.63
H100	4	allgather	262144	fp16	0.42
H100	4	allgather	524288	fp16	0.51
H100	4	allgather	1048576	fp16	0.54
H100	4	allgather	2097152	fp16	0.68
H100	4	allgather	4194304	fp16	1.35
H100	4	allgather	8388608	fp16	0.92
H100	4	allgather	16777216	fp16	0.80
H100	4	allgather	33554432	fp16	3.09
H100	4	allgather	67108864	fp16	5.33
H100	4	allgather	134217728	fp16	3.30
H100	4	allgather	268435456	fp16	6.49
H100	4	allgather	536870912	fp16	11.85
H100	4	allgather	1073741824	fp16	23.12
H100	4	allgather	2147483648	fp16	107.77
H100	4	allgather	4294967296	fp16	282.06
H100	4	allgather	8589934592	fp16	701.06
H100	8	allgather	32	fp16	0.14
H100	8	allgather	64	fp16	0.13
H100	8	allgather	128	fp16	0.22
H100	8	allgather	256	fp16	0.13
H100	8	allgather	512	fp16	0.20
H100	8	allgather	1024	fp16	0.12
H100	8	allgather	2048	fp16	0.21
H100	8	allgather	4096	fp16	0.18
H100	8	allgather	8192	fp16	0.14
H100	8	allgather	16384	fp16	0.17
H100	8	allgather	32768	fp16	0.14
H100	8	allgather	65536	fp16	1.46
H100	8	allgather	131072	fp16	0.86
H100	8	allgather	262144	fp16	0.57
H100	8	allgather	524288	fp16	0.72
H100	8	allgather	1048576	fp16	1.14
H100	8	allgather	2097152	fp16	1.16
H100	8	allgather	4194304	fp16	1.27
H100	8	allgather	8388608	fp16	1.37
H100	8	allgather	16777216	fp16	1.68
H100	8	allgather	33554432	fp16	2.50
H100	8	allgather	67108864	fp16	4.01
H100	8	allgather	134217728	fp16	7.65
H100	8	allgather	268435456	fp16	13.25
H100	8	allgather	536870912	fp16	25.87
H100	8	allgather	1073741824	fp16	109.26
H100	8	allgather	2147483648	fp16	304.80
H100	8	allgather	4294967296	fp16	969.17"""

df_network = pd.read_csv(StringIO(network_content), sep="\t")

# %%
df_network.head()

# %%
df_network.tail()

# %%
# Create a nested dictionary with (op, num_gpu) as the first key and nelem as the second key
network_time_dict = {}
for _, row in df_network.iterrows():
    key = (row['op'], row['num_gpu'])
    if key not in network_time_dict:
        network_time_dict[key] = {}
    network_time_dict[key][int(row['nelem'])] = row['latency(ms)']

# Sort the inner dictionaries by keys
for key, value in network_time_dict.items():
    network_time_dict[key] = {k: v for k, v in sorted(value.items())}

# Print the resulting dictionary
print(network_time_dict)

# %%
