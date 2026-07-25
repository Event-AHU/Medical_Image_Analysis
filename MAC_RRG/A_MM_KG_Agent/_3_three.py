import pandas as pd
import json

def normalize(s):
    return str(s).strip().lower()

def load_graph(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    return df

def extract_entity_links(df, entities, topk=10):
    result = {}
    for ent in entities.keys():
        ent_norm = normalize(ent)

        # 找出与 ent 相关的三元组
        mask = (df["source_entity"].str.lower() == ent_norm) | (df["target_entity"].str.lower() == ent_norm)
        sub = df.loc[mask].sort_values("count", ascending=False).head(topk)

        links = []
        for _, row in sub.iterrows():
            if normalize(row["source_entity"]) == ent_norm:
                links.append({"entity": row["target_entity"], "relation": row["type"], "count": int(row["count"])})
            else:
                links.append({"entity": row["source_entity"], "relation": row["type"], "count": int(row["count"])})

        result[ent] = links
    return result

if __name__ == "__main__":
    csv_path = "/data/qiaoyuhan/hhhh/Zretrival/kg_relations.csv"
    entities = {"lung": "anatomy", "pleural": "anatomy"}
    df = load_graph(csv_path)
    entity_links = extract_entity_links(df, entities, topk=10)

    with open("/data/qiaoyuhan/hhhh/Zretrival/entity_links.json", "w", encoding="utf-8") as f:
        json.dump(entity_links, f, ensure_ascii=False, indent=2)

    # # 打印 lung 的结果，每条一行横着显示
    # for item in entity_links["lung"]:
    #     print(json.dumps(item, ensure_ascii=False, separators=(',', ': ')))

    #     # 打印 lung 的结果，每条一行横着显示
    # for item in entity_links["pleural"]:
    #     print(json.dumps(item, ensure_ascii=False, separators=(',', ': ')))

    for ent, links in entity_links.items():
        print(f"=== {ent} ===")
        for item in links:
            print(json.dumps(item, ensure_ascii=False, separators=(',', ': ')))


