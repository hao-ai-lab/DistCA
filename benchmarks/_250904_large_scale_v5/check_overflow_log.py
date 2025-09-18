# %%
import os
import re
import pandas as pd
import glob

# Define the directory containing log files
log_dir = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250904_large_scale_v5/logs"

# List all experiment directories
exp_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

# Initialize lists to store extracted data
data = []

# Regular expression pattern to extract information
pattern = r'fs-mbz-gpu-(\d+).*:\🟡 \[Rank (\d+)\] Overflow check: ([\d.]+) GB, ([\d.]+) GB recv size, ([\d.]+) GB send last'

# Process each experiment directory
for exp_dir in exp_dirs:
    logs_path = os.path.join(log_dir, exp_dir, "logs")
    if not os.path.exists(logs_path):
        continue
    
    # Find all .out files
    out_files = glob.glob(os.path.join(logs_path, "*.out"))
    
    for file_path in out_files:
        # Extract node name from filename
        filename = os.path.basename(file_path)
        node_match = re.search(r'fs-mbz-gpu-(\d+)', filename)
        if not node_match:
            continue
        
        node_name = f"fs-mbz-gpu-{node_match.group(1)}"
        
        # Search for overflow check lines in the file
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
            
        matches = re.finditer(r'\🟡 \[Rank (\d+)\] Overflow check: ([\d.]+) GB, ([\d.]+) GB recv size, ([\d.]+) GB send last', content)
        
        for match in matches:
            rank = int(match.group(1))
            total_gb = float(match.group(2))
            recv_gb = float(match.group(3))
            send_gb = float(match.group(4))
            
            data.append({
                'experiment': exp_dir,
                'node': node_name,
                'rank': rank,
                'total_gb': total_gb,
                'recv_gb': recv_gb,
                'send_gb': send_gb
            })

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Optional: Save to CSV
df.to_csv('overflow_check_results.csv', index=False)
# %%
df.describe()
# %%
df.sort_values(by=['total_gb'], ascending=False)
# %%
