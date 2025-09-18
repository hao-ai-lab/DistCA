# %%

a = !ls ./logs/*/benchmark.raw.jsonl
b = !ls ./logs.v1/*/benchmark.raw.jsonl
# %%
c = a + b
# %%
c

# %%
mapping = {}

all_rows = []
import json
for file in c:
    with open(file, 'r') as f:
        data = []
        for line in f:
            row = json.loads(line)
            row['name'] = file
            data.append(row)
            all_rows.append(row)
    print(data)
    mapping[file] = data

# %%
import pandas as pd
df = pd.DataFrame(all_rows)

# %%
df
# %%
df.name = df.name.str.replace("logs/", "").str.replace("./logs.v1/", "").str.replace(".*PST_", "", regex=True).str.replace("/benchmark.raw.jsonl", "")
# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df
# %%
