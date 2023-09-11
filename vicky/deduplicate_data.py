from typing import List, Tuple, Dict, Any

import numpy as np


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> Dict[str, Any]:
    a, b = permutations
    masks: np.ndarray = np.full(shape=num_perm, dtype=np.uint64, fill_value=MAX_HASH)
    tokens: Set[str] = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)}
    hashvalues: np.ndarray = np.array([sha1_hash(token.encode("utf-8")) for token in tokens], dtype=np.uint64)
    permuted_hashvalues = np.bitwise_and(
        ((hashvalues * np.tile(a, (len(hashvalues), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )
    hashvalues = np.vstack([permuted_hashvalues, masks]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


embedded = ds.map(
	function=embed_func,
	fn_kwargs={
		"num_perm": args.num_perm,
		"hashranges": HASH_RANGES,
		"ngram_size": args.ngram,
		"permutations": PERMUTATIONS,
	},
	input_columns=[args.column],
	remove_columns=ds.column_names,
	num_proc=os.cpu_count(),
	with_indices=True,
	desc="Fingerprinting...",
)


for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
	for cluster in table.values():
		if len(cluster) <= 1:
			continue
		idx = min(cluster)
		for x in cluster:
			uf.union(x, idx)



edges = (
	records.flatMap(
		lambda x: generate_hash_values(
			content=x[1],
			idx=x[0],
			num_perm=args.num_perm,
			ngram_size=args.ngram_size,
			hashranges=HASH_RANGES,
			permutations=PERMUTATIONS,
		)
	)
	.groupBy(lambda x: (x[0], x[1]))
	.flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
	.distinct()
	.cache()
)

a = edges
while True:
	b = a.flatMap(large_star_map).groupByKey().flatMap(large_star_reduce).distinct().cache()
	a = b.map(small_star_map).groupByKey().flatMap(small_star_reduce).distinct().cache()
	changes = a.subtract(b).union(b.subtract(a)).collect()
	if len(changes) == 0:
		break

results = a.collect()
