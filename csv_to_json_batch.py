import pandas as pd
import json

df = pd.read_csv("model_dataset_v2.csv")

n1 = 15
n2 = 25

batch = []

for idx, i in enumerate(range(n1, n2 + 1), start=1):
    
    sample_row = df.iloc[i].to_dict()

    sample_row.pop('is_late', None)

    new_json = {
        "order_id": f"test_order_{idx}",
        **sample_row
    }

    batch.append(new_json)

print(json.dumps(batch, indent=4, ensure_ascii=False))