"""DomainNet parquet → 图片目录 + PACS 风格 txt 划分

前置: downloads/domainnet_parquet/ 下有 4 个 parquet 分片;
      downloads/domainnet_readme.md 为数据集 README (含 345 类名与 6 域名的权威列表)。

产出:
  data/domainnet/images/<domain>/<class>/<file>
  data/domainnet/dn_real_train.txt / dn_real_test.txt / dn_painting_test.txt (relpath,label)
  data/domainnet/dn_classes.txt / dn_concepts.txt
"""
import io
import os
import re
import glob

import pyarrow.parquet as pq

SRC_DIR = "downloads/domainnet_parquet"
README = os.path.join("downloads", "domainnet_readme.md")
OUT_ROOT = os.path.join("data", "domainnet", "images")
OUT_ANNO = os.path.join("data", "domainnet")
KEEP_DOMAINS = {"real", "painting"}   # 当前协议: Real→Painting 单域对; 其余域可选补提


def parse_readme():
    t = open(README, encoding="utf-8").read()
    i = t.find("- name: label")
    label_block = t[i: t.find("- name:", i + 10)]
    classes = re.findall(r"'\d+': ([A-Za-z_0-9]+)", label_block)
    j = t.find("- name: domain")
    dom_block = t[j: j + 3000]
    domains = re.findall(r"'\d+': ([a-z_]+)", dom_block)
    return classes, domains


def main():
    classes, domain_names = parse_readme()
    print(f"345 类名 {len(classes)} 个 | 域名 {domain_names}")
    assert len(classes) == 345 and len(domain_names) == 6
    dom_i2n = {i: n for i, n in enumerate(domain_names)}

    os.makedirs(OUT_ANNO, exist_ok=True)
    splits = {"real_train": [], "real_test": [], "painting_test": []}
    counters = {k: 0 for k in splits}
    n_bad = 0

    shards = sorted(glob.glob(os.path.join(SRC_DIR, "*.parquet")))
    assert shards, "未找到 parquet 分片"

    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label", "domain"]):
            imgs = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            doms = batch.column("domain").to_pylist()
            for i in range(len(labels)):
                dn_name = dom_i2n.get(doms[i])
                if dn_name not in KEEP_DOMAINS:
                    continue
                if "train" in shard:
                    key = "real_train" if dn_name == "real" else None
                else:
                    key = ("real_test" if dn_name == "real"
                           else "painting_test" if dn_name == "painting" else None)
                if key is None:
                    continue
                rec = imgs[i]
                data_bytes = rec.get("bytes") if isinstance(rec, dict) else None
                if not data_bytes:
                    n_bad += 1
                    continue
                cls_name = classes[labels[i]]
                ddir = os.path.join(OUT_ROOT, dn_name, cls_name)
                os.makedirs(ddir, exist_ok=True)
                base = os.path.basename(rec.get("path") or f"img_{counters[key]:07d}.jpg")
                if not base.lower().endswith((".jpg", ".jpeg", ".png")):
                    base += ".jpg"
                counters[key] += 1
                fname = f"{counters[key]:07d}_{base}"
                with open(os.path.join(ddir, fname), "wb") as f:
                    f.write(data_bytes)
                splits[key].append(f"{dn_name}/{cls_name}/{fname},{labels[i]}")
        print(f"[done] {os.path.basename(shard)}", flush=True)

    for key, rows in splits.items():
        with open(os.path.join(OUT_ANNO, f"dn_{key}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        print(f"dn_{key}.txt: {len(rows)} 行")

    open(os.path.join(OUT_ANNO, "dn_classes.txt"), "w", encoding="utf-8").write(
        "\n".join(f"{i} {c}" for i, c in enumerate(classes)))
    # 概念集: 345 类名(下划线→空格) + 通用视觉属性, 去重
    generic = [c.strip() for c in
               open("data/pacs/pacs_concepts.txt", encoding="utf-8").read().replace("\n", ",").split(",")
               if c.strip()]
    seen, con = set(), []
    for cname in classes:
        cc = cname.replace("_", " ")
        if cc not in seen:
            seen.add(cc); con.append(cc)
    for g in generic:
        if g not in seen:
            seen.add(g); con.append(g)
    open(os.path.join(OUT_ANNO, "dn_concepts.txt"), "w", encoding="utf-8").write("\n".join(con))
    print(f"概念集: {len(con)} 个 | 无效样本: {n_bad}")


if __name__ == "__main__":
    main()
