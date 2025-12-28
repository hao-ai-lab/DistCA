# from datasets import load_dataset

# train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train')
# train_data.select(range(10000)).to_json("codeparrot_data.json", lines=True)


from datasets import load_dataset

N = 100000

train_data = load_dataset(
    'codeparrot/codeparrot-clean-train', split=f'train[:{N}]'
)
train_data.to_json("codeparrot_data.json", lines=True)