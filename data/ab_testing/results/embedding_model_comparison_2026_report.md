# A/B Test Results: embedding_model_comparison_2026

Comparison of embedding models for metadata retrieval quality

**Generated:** 2026-04-09 14:45:20

## Metadata

- **experiment_name:** embedding_model_comparison_2026
- **num_variants:** 5

## Metrics

### Variants

#### Bge Large

##### Precision At K

- **1:** 0.5000
- **10:** 0.1656
- **3:** 0.3854
- **5:** 0.3312

##### Recall At K

- **1:** 0.1698
- **10:** 0.4771
- **3:** 0.3484
- **5:** 0.4771

- **mrr:** 0.6089
- **map:** 0.3789
- **ndcg:** 0.4695
##### Metadata

- **num_queries:** 32
- **aggregation:** mean


#### Bge Small

##### Precision At K

- **1:** 0.5000
- **10:** 0.1625
- **3:** 0.4167
- **5:** 0.3250

##### Recall At K

- **1:** 0.1458
- **10:** 0.4484
- **3:** 0.3563
- **5:** 0.4484

- **mrr:** 0.6172
- **map:** 0.3581
- **ndcg:** 0.4582
##### Metadata

- **num_queries:** 32
- **aggregation:** mean


#### E5 Large

##### Precision At K

- **1:** 0.5312
- **10:** 0.1625
- **3:** 0.4062
- **5:** 0.3250

##### Recall At K

- **1:** 0.1641
- **10:** 0.4312
- **3:** 0.3297
- **5:** 0.4312

- **mrr:** 0.6130
- **map:** 0.3454
- **ndcg:** 0.4454
##### Metadata

- **num_queries:** 32
- **aggregation:** mean


#### Gte Large

##### Precision At K

- **1:** 0.5312
- **10:** 0.1656
- **3:** 0.3958
- **5:** 0.3312

##### Recall At K

- **1:** 0.1641
- **10:** 0.4677
- **3:** 0.3401
- **5:** 0.4677

- **mrr:** 0.6276
- **map:** 0.3741
- **ndcg:** 0.4708
##### Metadata

- **num_queries:** 32
- **aggregation:** mean


#### Baseline Mpnet

##### Precision At K

- **1:** 0.6562
- **10:** 0.2562
- **3:** 0.6042
- **5:** 0.5125

##### Recall At K

- **1:** 0.2240
- **10:** 0.7500
- **3:** 0.5547
- **5:** 0.7500

- **mrr:** 0.7510
- **map:** 0.6409
- **ndcg:** 0.7116
##### Metadata

- **num_queries:** 32
- **aggregation:** mean



