# HopfE: Knowledge Graph Representation Learning using Inverse Hopf Fibrations

This repository is the official pytorch implementation of HopfE (https://arxiv.org/abs/2108.05774).

Recently, several Knowledge Graph Embedding (KGE) approaches have been devised to represent entities and relations in dense vector space and employed in downstream tasks such as link prediction. A few KGE techniques address interpretability, i.e., mapping the connectivity patterns of the relations (i.e., symmetric/asymmetric, inverse, and composition) to a geometric interpretation such as rotations. Other approaches model the representations in higher dimensional space such as four-dimensional space (4D) to enhance the ability to infer the connectivity patterns (i.e., expressiveness). However, modeling relation and entity in a 4D space often comes at the cost of interpretability. This paper proposes HopfE, a novel KGE approach aiming to achieve the interpretability of inferred relations in the four-dimensional space. We first model the structural embeddings in 3D Euclidean space and view the relation operator as an SO(3) rotation. Next, we map the entity embedding vector from a 3D space to a 4D hypersphere using the inverse Hopf Fibration, in which we embed the semantic information from the KG ontology. Thus, HopfE considers the structural and semantic properties of the entities without losing expressivity and interpretability. Our empirical results on four well-known benchmarks achieve state-of-the-art performance for the KG completion task.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
wn18: bash run.sh train HopfE wn18 <gpu-id> <dir> 512 1024 200 12.0 0.3 0.1 18000 8 0 -me -mr -adv
wn18rr: bash run.sh train HopfE wn18rr <gpu-id> <dir> 512 512 200 6.0 0.5 0.1 15000 8 0 -me -mr -adv
FB15k-237: bash run.sh train HopfE FB15k-237 <gpu-id> <dir> 1024 256 500 9.0 1.0 0.1 27000 16 0 -me -mr -adv 
YAGO3-10: bash run.sh train HopfE YAGO3-10 <gpu-id> <dir> 1024 512 200 24.0 1.0 0.1 33000 16 0 -me -mr -adv
```
## Evaluation

To evaluate run:

```eval
wn18: bash run.sh test HopfE wn18 <gpu-id> <dir> 512 1024 200 12.0 0.3 0.1 18000 8 0 -me -mr -adv
wn18rr: bash run.sh test HopfE wn18rr <gpu-id> <dir> 512 512 200 6.0 0.5 0.1 15000 8 0 -me -mr -adv
FB15k-237: bash run.sh test HopfE FB15k-237 <gpu-id> <dir> 1024 256 500 9.0 1.0 0.1 27000 16 0 -me -mr -adv
YAGO3-10: bash run.sh test HopfE YAGO3-10 <gpu-id> <dir> 1024 512 200 24.0 1.0 0.1 33000 16 0 -me -mr -adv
```


<!-- ## Results

Our model achieves the following performance on:

WN18

| Model name | MR | MRR | H@1 | H@3 | H@10|
| ---------- |----|-----|-----|-----|-----|
|  RotatE    |309 |0.949|0.944|0.952|0.959|
|   QuatE	   |388	|0.949|0.941|0.954|0.960|
|   DensE    |285	|0.950|0.945|0.954|0.959|


WN18RR

| Model name | MR | MRR | H@1 | H@3 | H@10|
| ---------- |----|-----|-----|-----|-----|
|   RotatE   |3340|0.476|0.428|0.492|0.571|
|   QuatE    |3472|0.481|0.436|0.500|0.564|
|   DensE    |3052|0.491|0.443|0.508|0.579|

FB15k-237

| Model name | MR | MRR | H@1 | H@3 | H@10|
| ---------- |----|-----|-----|-----|-----|
|   RotatE	 |177	|0.338|0.241|0.375|0.533|
|   QuatE    |176	|0.311|0.221|0.342|0.495|
|   DensE    |169	|0.349|0.256|0.384|0.535|


YAGO3-10

| Model name | MR | MRR | H@1 | H@3 | H@10|
| ---------- |----|-----|-----|-----|-----|
|  RotatE    |1767|0.495|0.402|0.550|0.670|
|   DensE    |1450|0.541|0.465|0.585|0.678| -->

<!-- ## Contributing

This respoisitory is a open source software under MIT lisence. If you'd like to contribute, or have any suggestions for this project, please open an issue on this GitHub repository. -->

# Citation
If you use our work kindly consider citing

```
@inbook{10.1145/3459637.3482263,
author = {Bastos, Anson and Singh, Kuldeep and Nadgeri, Abhishek and Shekarpour, Saeedeh and Mulang, Isaiah Onando and Hoffart, Johannes},
title = {HopfE: Knowledge Graph Representation Learning Using Inverse Hopf Fibrations},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3482263},
abstract = {Recently, several Knowledge Graph Embedding (KGE) approaches have been devised to represent entities and relations in a dense vector space and employed in downstream tasks such as link prediction. A few KGE techniques address interpretability, i.e., mapping the connectivity patterns of the relations (symmetric/asymmetric, inverse, and composition) to a geometric interpretation such as rotation. Other approaches model the representations in higher dimensional space such as four-dimensional space (4D) to enhance the ability to infer the connectivity patterns (i.e., expressiveness). However, modeling relation and entity in a 4D space often comes at the cost of interpretability. We propose HopfE, a novel KGE approach aiming to achieve the interpretability of inferred relations in the four-dimensional space. HopfE models the structural embeddings in 3D Euclidean space. Next, we map the entity embedding vector from a 3D Euclidean space to a 4D hypersphere using the inverse Hopf Fibration, in which we embed the semantic information from the KG ontology. Thus, HopfE considers the structural and semantic properties of the entities without losing expressivity and interpretability. Our empirical results on four well-known benchmarks achieve state-of-the-art performance for KG completion.},
booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
pages = {89–99},
numpages = {11}
}
```

## Acknowledgement 

The evaluation code is implemented based on the open source code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
