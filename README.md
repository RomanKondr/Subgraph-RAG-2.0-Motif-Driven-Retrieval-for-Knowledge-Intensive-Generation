# SubgraphRAG 2.0: Motif-Driven Retrieval for Knowledge-Intensive Generation

This repository extends **SubgraphRAG** with motif-aware retrieval methods for knowledge-graph-based retrieval-augmented generation under a fixed prompt budget.

[[Original SubgraphRAG Paper]](https://arxiv.org/abs/2410.20724)

![model](framework_241104.png)

## Table of Contents

- [Overview](#overview)
- [What This Fork Adds](#what-this-fork-adds)
- [Main Results](#main-results)
- [Key Takeaway](#key-takeaway)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [Citation](#citation)

## Overview

SubgraphRAG is a retrieval-and-reasoning pipeline for knowledge-graph-based retrieval-augmented generation.  
This fork studies whether **higher-order graph motifs** such as **wedges** and **triangles** can improve KGQA retrieval when evidence must fit into a strict top-100 prompt budget.

The main question explored in this project is:

> Can motif-derived evidence improve retrieval quality without crowding out the strong high-precision baseline triples already selected by SubgraphRAG?

Rather than treating motifs as raw graph expansions, this project tests **budgeted, query-conditioned motif integration**.

## What This Fork Adds

This fork extends the original SubgraphRAG pipeline with several motif-aware retrieval variants:

### 1. Motif integration regimes
- **REPLACE**  
  Replace baseline-ranked triples with motif-expanded triples.
- **AUGMENT / closure**  
  Keep the baseline evidence and append motif-derived triples.
- **Motif-units + quota packing**  
  Treat motifs as explicit retrieval units, score them, expand them into triples, and reserve a fixed quota of prompt slots for motif-derived evidence.

### 2. Question anchoring
- Restricts motif influence to neighborhoods around question-linked entities.
- Supports both:
  - strict anchor membership
  - one-hop anchor neighborhoods

### 3. Motif-based reranking
- Adds bounded motif bonuses to top-ranked triples.
- Evaluates both:
  - triangle-based motif bonuses
  - wedge-based motif bonuses

### 4. Budget-aware prompt packing
- Forces motif-derived triples into the visible top-100 prompt window.
- Tests how motif evidence competes with baseline triples under fixed prompt constraints.

### 5. Evaluation improvements
- **Matched-ID evaluation** to ensure fair comparisons across the same question IDs and ordering
- **Prompt visibility checks** to verify that retrieval changes actually alter the final evidence shown to the LLM

## Main Results

The strongest result came from **motif-units + quota packing**, where motif-derived triples were integrated as a controlled secondary evidence channel rather than replacing the baseline.

### Core comparison

| Method | Hit@1 | MacroF1 | EM | TW | no-ans |
|---|---:|---:|---:|---:|---:|
| Baseline | 90.6 | 61.45 | 41.6 | 11.2 | 15.2% |
| Motif REPLACE | 43.6 | 29.63 | 19.8 | 56.4 | 47.8% |
| Motif AUGMENT / closure | 88.6 | 60.12 | 40.8 | 11.4 | 14.6% |
| Motif-units packing (quota = 35) | **92.4** | **62.82** | **42.6** | **7.6** | 14.6% |

### Quota ablation

| Quota Q | Hit@1 | MacroF1 | EM | TW | no-ans |
|---|---:|---:|---:|---:|---:|
| 0 (Baseline) | 90.6 | 61.45 | 41.6 | 11.2 | 15.2% |
| 10 | 89.8 | 61.12 | 41.4 | 10.1 | 14.7% |
| 35 | **92.4** | **62.82** | **42.6** | **7.6** | 14.6% |
| 60 | 88.2 | 59.65 | 40.3 | 12.4 | 15.4% |

### Main findings
- **Naively replacing** baseline evidence with motif-expanded triples is harmful.
- **Simple augmentation** is mostly near-neutral under a fixed budget.
- **Moderate motif budget** helps.
- **Large motif budget** hurts by crowding out high-precision baseline evidence.
- Motifs are useful only when integrated as **budgeted, query-conditioned evidence units**.

## Key Takeaway

The main lesson from this work is that motifs do **not** help simply because they add more structure.  
They help only when they are:

- **query-conditioned**
- **budget-controlled**
- **explicitly packed into the top-B prompt window**

Under a fixed evidence budget, **packing is part of retrieval quality**.

## Repository Structure

This fork keeps the original retrieval-and-reasoning pipeline structure:

- [`retrieve/`](./retrieve/) — retrieval stage
- [`reason/`](./reason/) — reasoning / generation stage

Motif-aware retrieval changes are implemented on top of this baseline workflow.

## Usage

This repository follows the original SubgraphRAG pipeline structure.

1. For the retrieval stage, see [the retrieve folder](./retrieve/).
2. For the reasoning stage, see [the reason folder](./reason/).

The main difference in this fork is the addition of motif-aware retrieval variants, including:

- motif-based reranking
- question-anchored motif scoring
- motif-unit scoring
- quota-based prompt packing


## Citation

If you use the original framework, please cite:

```tex
@inproceedings{li2024subgraphrag,
    title={Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation},
    author={Li, Mufei and Miao, Siqi and Li, Pan},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
