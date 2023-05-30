from my_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
target = item["target"]
txt = item["caption"]
hint = item["hint"]
print(txt)
print(target.shape)
print(hint.shape)
