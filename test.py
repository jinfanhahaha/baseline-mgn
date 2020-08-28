import json
import numpy as np
import time


def cos_sim(v_a, v_b):
    v_a = np.mat(v_a)
    v_b = np.mat(v_b)
    num = float(v_a * v_b.T)
    denom = np.linalg.norm(v_a) * np.linalg.norm(v_b)
    cos = num / denom
    sim = 0.5 + 0.5*cos
    return 1 / sim


def two_D(v_a, v_b):
    return np.sqrt(np.sum(np.square(np.array(v_a) - np.array(v_b))))


def one_D(v_a, v_b):
    return np.sum(np.abs(np.array(v_a) - np.array(v_b)))


with open("/raid/ours_data/Competition/NAIC-ReID/gallery_features.json",
          'r', encoding='utf-8') as f:
    gallery_features = json.load(f)


with open("/raid/ours_data/Competition/NAIC-ReID/query_features.json",
          'r', encoding='utf-8') as f:
    query_features = json.load(f)

keys = sorted(list(query_features.keys()))
res_dict = {}
t = 0
for q_k in keys:
    test_query = query_features[q_k]
    ids = []
    sims = []
    for g_k in gallery_features.keys():
        g_value = gallery_features[g_k]
        sims.append(cos_sim(test_query, g_value))
        ids.append(g_k)
    ids = np.array(ids)
    sims = np.array(sims)
    index = np.argsort(sims)
    ids = ids[index]
    sims = sims[index]
    res_dict[q_k] = ids[:200].tolist()
    if t == 5:
        break
    t += 1

with open("./result/test_cos.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(res_dict))

