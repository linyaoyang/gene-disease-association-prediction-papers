# Gene-disease-association-prediction-papers
A collection of papers and resources about gene disease association prediction.

## Table of Contents
- [Gene-disease-association-prediction-papers](#Gene-disease-association-prediction-papers)
  - [Related Surveys](#related-surveys)
  - [Datasets](#Datasets)
    - [Gene-related datasets](#gene-related-datasets)
    - [Disease-related datasets](#disease-related-datasets)
    - [Other datasets](#other-datasets)
  - [Methods](#methods)
    - [Network-based methods](#network-based-methods)
    - [Machine learning methods](#machine-learning-methods)
    - [Deep learning methods](#deep-learning-methods)
    - [Graph neural networks](#graph-neural-networks)
    - [Large language models](#large-language-models)
  - [Advances and Applications](#advances-and-applications)
    - [Evaluation protocols and Results](#evaluation-protocols-and-results)
    - [Applications](#applications)
  - [Challenges and Future Directions](#challenges-and-future-directions)

## Related Surveys

* Unifying Large Language Models and Knowledge Graphs: A Roadmap (TKDE, 2024) [[paper]](https://arxiv.org/pdf/2306.08302.pdf)
* A Survey on Knowledge-Enhanced Pre-trained Language Models (Arxiv, 2023) [[paper]](https://arxiv.org/pdf/2212.13428.pdf)
* A Survey of Knowledge-Intensive NLP with Pre-Trained Language Models (Arxiv, 2022) [[paper]](https://arxiv.org/pdf/2202.08772.pdf)
* A Review on Language Models as Knowledge Bases (Arxiv, 2022) [[paper]](https://arxiv.org/pdf/2204.06031.pdf)
* Generative Knowledge Graph Construction: A Review (EMNLP, 2022) [[paper]](https://arxiv.org/pdf/2210.12714.pdf)
* Knowledge Enhanced Pretrained Language Models: A Compreshensive Survey (Arxiv, 2021) [[paper]](https://arxiv.org/pdf/2110.08455.pdf)
* Reasoning over Different Types of Knowledge Graphs: Static, Temporal and Multi-Modal (Arxiv, 2022) [[paper]](https://arxiv.org/abs/2212.05767)[[code]](https://github.com/LIANGKE23/Awesome-Knowledge-Graph-Reasoning)

## Datasets
### Gene-related datasets
- ERNIE: Enhanced Language Representation with Informative Entities (ACL, 2019) [[paper]](https://aclanthology.org/P19-1139.pdf)
- Exploiting structured knowledge in text via graph-guided representation learning (EMNLP, 2019) [[paper]](https://aclanthology.org/2020.emnlp-main.722.pdf)
- SKEP: Sentiment knowledge enhanced pre-training for sentiment analysis (ACL, 2020) [[paper]](https://aclanthology.org/2020.acl-main.374.pdf)
- E-bert: A phrase and product knowledge enhanced language model for e-commerce (Arxiv, 2020) [[paper]](https://arxiv.org/pdf/2009.02835.pdf)
- Pretrained encyclopedia: Weakly supervised knowledge-pretrained language model (ICLR, 2020) [[paper]](https://openreview.net/pdf?id=BJlzm64tDH)
- BERT-MK: Integrating graph contextualized knowledge into pre-trained language models (EMNLP, 2020) [[paper]](https://aclanthology.org/2020.findings-emnlp.207.pdf)
- K-BERT: enabling language representation with knowledge graph (AAAI, 2020) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5681/5537)
- CoLAKE: Contextualized language and knowledge embedding (COLING, 2020) [[paper]](https://aclanthology.org/2020.coling-main.327.pdf)
- Kepler: A unified model for knowledge embedding and pre-trained language representation (TACL, 2021) [[paper]](https://aclanthology.org/2021.tacl-1.11.pdf)
- K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters (ACL Findings, 2021) [[paper]](https://aclanthology.org/2021.findings-acl.121.pdf)
- Cokebert: Contextual knowledge selection and embedding towards enhanced pre-trained language models (AI Open, 2021) [[paper]](https://www.sciencedirect.com/science/article/pii/S2666651021000188/pdfft?md5=75919f85dcb5711fd2fe9e3785b24982&pid=1-s2.0-S2666651021000188-main.pdf)
- Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation (Arixv, 2021) [[paper]](https://arxiv.org/pdf/2107.02137)
- Pre-training language models with deterministic factual knowledge (EMNLP, 2022) [[paper]](https://aclanthology.org/2022.emnlp-main.764.pdf)
- Kala: Knowledge-augmented language model adaptation (NAACL, 2022) [[paper]](https://aclanthology.org/2022.naacl-main.379.pdf)
- DKPLM: decomposable knowledge-enhanced pre-trained language model for natural language understanding (AAAI, 2022) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21425/21174)
- Dict-BERT: Enhancing language model pre-training with dictionary (ACL Findings, 2022) [[paper]](https://aclanthology.org/2022.findings-acl.150.pdf)
- JAKET: joint pre-training of knowledge graph and language understanding (AAAI, 2022) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21417/21166)
- Tele-Knowledge Pre-training for Fault Analysis (ICDE, 2023) [[paper]](https://arxiv.org/abs/2210.11298)

### Disease-related datasets
- Barack’s wife hillary: Using knowledge graphs for fact-aware language modeling (ACL, 2019) [[paper]](https://aclanthology.org/P19-1598.pdf)
- Retrieval-augmented generation for knowledge-intensive nlp tasks (NeurIPS, 2020) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
- Realm: Retrieval-augmented language model pre-training (ICML, 2020) [[paper]](https://dl.acm.org/doi/pdf/10.5555/3524938.3525306)
- QA-GNN: Reasoning with language models and knowledge graphs for question answering (NAACL, 2021) [[paper]](https://aclanthology.org/2021.naacl-main.45.pdf)
- Memory and knowledge augmented language models for inferring salience in long-form stories (EMNLP, 2021) [[paper]](https://aclanthology.org/2021.emnlp-main.65.pdf)
- JointLK: Joint reasoning with language models and knowledge graphs for commonsense question answering (NAACL, 2022) [[paper]](https://aclanthology.org/2022.naacl-main.372.pdf)
- Enhanced Story Comprehension for Large Language Models through Dynamic Document-Based Knowledge Graphs (AAAI, 2022) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21286)
- Greaselm: Graph reasoning enhanced language models (ICLR, 2022) [[paper]](https://openreview.net/pdf?id=41e9o6cQPj)
- An efficient memory-augmented transformer for knowledge-intensive NLP tasks (EMNLP, 2022) [[paper]](https://aclanthology.org/2022.emnlp-main.346.pdf)
- Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering (NLRSE@ACL, 2023) [[paper]](https://arxiv.org/abs/2306.04136)
- LLM-Based Multi-Hop Question Answering with Knowledge Graph Integration in Evolving Environments (EMNLP Findings, 2024) [[paper](https://arxiv.org/abs/2408.15903)] 


### Other datasets
- Language models as knowledge bases (EMNLP, 2019) [[paper]](https://arxiv.org/pdf/1909.01066.pdf)
- Kagnet: Knowledge-aware graph networks for commonsense reasoning (Arxiv, 2019) [[paper]](https://arxiv.org/pdf/1909.02151.pdf)
- Autoprompt: Eliciting knowledge from language models with automatically generated prompts (EMNLP, 2020) [[paper]](https://arxiv.org/pdf/2010.15980.pdf)
- How can we know what language models know? (ACL, 2020) [[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96460)
- Knowledge neurons in pretrained transformers (ACL, 2021) [[paper]](https://arxiv.org/pdf/2104.08696.pdf)
- Can Language Models be Biomedical Knowledge Bases? (EMNLP, 2021) [[paper]](https://arxiv.org/pdf/2109.07154.pdf)
- Interpreting language models through knowledge graph extraction (Arxiv, 2021) [[paper]](https://arxiv.org/pdf/2111.08546.pdf)
- QA-GNN: Reasoning with language models and knowledge graphs for question answering (ACL, 2021) [[paper]](https://arxiv.org/pdf/2104.06378.pdf)
- How to Query Language Models? (Arxiv, 2021) [[paper]](https://arxiv.org/pdf/2108.01928.pdf)
- Rewire-then-probe: A contrastive recipe for probing biomedical knowledge of pre-trained language models (Arxiv, 2021) [[paper]](https://arxiv.org/pdf/2110.08173.pdf)
- When Not to Trust Language Models: Investigating Effectiveness and Limitations of Parametric and Non-Parametric Memories (Arxiv, 2022) [[paper]](https://arxiv.org/pdf/2212.10511.pdf)
- How Pre-trained Language Models Capture Factual Knowledge? A Causal-Inspired Analysis (Arxiv, 2022) [[paper]](https://arxiv.org/pdf/2203.16747.pdf)
- Can Knowledge Graphs Simplify Text? (CIKM, 2023) [[paper]](https://dl.acm.org/doi/10.1145/3583780.3615514)

## Methods
### Network-based methods
- Network-based approaches for disease-gene association prediction using protein-protein interaction networks (International Journal of Molecular Sciences, 2022) [[paper]](https://www.mdpi.com/1422-0067/23/13/7411)
- Predicting Drug–Gene–Disease Associations by Tensor Decomposition for Network-Based Computational Drug Repositioning (Biomedicines, 2023) [[paper]](https://www.mdpi.com/1422-0067/23/13/7411)
- Identifying potential association on gene-disease network via dual hypergraph regularized least squares (BMC Genomics, 2021) [[paper]](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-021-07864-z) [[Code]](https://github.com/guofei-tju/DHRLS)
- Gene–disease association with literature based enrichment (Journal of Biomedical Informatics, 2014) [[paper]](https://www.sciencedirect.com/science/article/pii/S1532046414000641?via%3Dihub)
- Recent advances in network-based methods for disease gene prediction (Briefings in Bioinformatics, 2021) [[paper]](https://academic.oup.com/bib/article/22/4/bbaa303/6023077?login=false)
- Prediction and Validation of Gene-Disease Associations Using Methods Inspired by Social Network Analyses (Plos One, 2013) [[paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0058977)
- ANTENNA, a Multi-Rank, Multi-Layered Recommender System for Inferring Reliable Drug-Gene-Disease Associations: Repurposing Diazoxide as a Targeted Anti-Cancer Therapy (IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2018) [[paper]](https://ieeexplore.ieee.org/document/8318603)
- FocusHeuristics – expression-data-driven network optimization and disease gene prediction (Scientific Reports, 2017) [[paper]](https://www.nature.com/articles/srep42638)
- A knowledge-based approach for predicting gene–disease associations (Bioinformatics, 2016) [[paper]](https://academic.oup.com/bioinformatics/article/32/18/2831/1744378) [[Code]](http://cssb2.biology.gatech.edu/knowgene)
- The research on gene-disease association based on text-mining of PubMed (BMC Bioinformatics, 2018) [[paper]](https://link.springer.com/article/10.1186/s12859-018-2048-y) [[Code]](https://github.com/jiezhou1111/The-Research-on-Gene-Disease-Association-Based-on-Text-Mining-of-PubMed)
- Inductive matrix completion for predicting gene–disease associations (Bioinformatics, 2014) [[paper]](https://academic.oup.com/bioinformatics/article/30/12/i60/385272)
- Gene co-expression analysis for functional classification and gene–disease predictions (Briefings in Bioinformatics, 2018) [[paper]](https://academic.oup.com/bib/article/19/4/575/2888441)
- Identifying gene-disease associations using centrality on a literature mined gene-interaction network (Bioinformatics, 2008) [[paper]](https://academic.oup.com/bioinformatics/article/24/13/i277/236041)
- Disease gene prediction for molecularly uncharacterized diseases (PLoS computational biology, 2019) [[paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007078)
- Network-based association analysis to infer new disease-gene relationships using large-scale protein interactions (PLoS One, 2018) [[paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199435)
- Prediction and Validation of Disease Genes Using HeteSim Scores (IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2017) [[paper]](https://ieeexplore.ieee.org/abstract/document/7406753?casa_token=DmIQoe0OXVkAAAAA:rMD-fktnFocbRaLvmXWoRzdOkn93-auoG0-hyo_HvXU0rywJGowpHE-4Rh_NxSJGQqv5YODymld5) [[Code]](http://lab.malab.cn/data/HeteSim/index.jsp)
- DGLinker: flexible knowledge-graph prediction of disease–gene associations (Nucleic Acids Research, 2021) [[paper]](https://academic.oup.com/nar/article/49/W1/W153/6298614?login=false) [[Code]](https://dglinker.rosalind.kcl.ac.uk)
- Multimodal network diffusion predicts future disease–gene–chemical associations (Bioinformatics, 2019) [[paper]](https://academic.oup.com/bioinformatics/article/35/9/1536/5124277?searchresult=1) [[Code]](https://github.com/LichtargeLab/multimodal-network-diffusion)
- Gene co-expression analysis for functional classification and gene–disease predictions (Briefings in Bioinformatics, 2018) [[paper]](https://academic.oup.com/bib/article/19/4/575/2888441?searchresult=1)
- NIDM: network impulsive dynamics on multiplex biological network for disease-gene prediction (Briefings in Bioinformatics, 2021) [[paper]](https://academic.oup.com/bib/article/22/5/bbab080/6236070?searchresult=1)
- HyMM: hybrid method for disease-gene prediction by integrating multiscale module structure (Briefings in Bioinformatics, 2022) [[paper]](https://academic.oup.com/bib/article/23/3/bbac072/6547263?searchresult=1)
- A network-based method for brain disease gene prediction by integrating brain connectome and molecular network (Briefings in Bioinformatics, 2022) [[paper]](https://academic.oup.com/bib/article/23/1/bbab459/6415315?searchresult=1) [[Code]](https://github.com/MedicineBiology-AI/brainMI)

### Machine learning methods
- Analysis for Disease Gene Association Using Machine Learning (IEEE Access, 2020) [[paper]](https://ieeexplore.ieee.org/abstract/document/9181557#full-text-header)
- Probability-based collaborative filtering model for predicting gene–disease associations (BMC Medical Genomics, 2017) [[paper]](https://link.springer.com/article/10.1186/s12920-017-0313-y)
- Automatic extraction of gene-disease associations from literature using joint ensemble learning (PLos One, 2018) [[paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0200699)
- A feature-learning-based method for the disease-gene prediction problem (International Journal of Data Mining and Bioinformatics, 2020) [[paper]](https://www.inderscienceonline.com/doi/epdf/10.1504/IJDMB.2020.109502)
- Group spike-and-slab lasso generalized linear models for disease prediction and associated genes detection by incorporating pathway information (Bioinformatics, 2018) [[paper]](https://academic.oup.com/bioinformatics/article/34/6/901/4565593?searchresult=1) [[Code]](http://www.ssg.uab.edu/bhglm/)
- DiSTect: a Bayesian model for disease-associated gene discovery and prediction in spatial transcriptomics (Bioinformatics, 2025) [[paper]](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaf530/8261372?searchresult=1) [[Code]](https://github.com/StaGill/DiSTect)
- AITeQ: a machine learning framework for Alzheimer’s prediction using a distinctive five-gene signature (Briefings in Bioinformatics, 2024) [[paper]](https://academic.oup.com/bib/article/25/4/bbae291/7692307?searchresult=1) [[Code]](https://github.com/ishtiaque-ahammad/AITeQ)

### Deep learning methods
- PheSeq, a Bayesian deep learning model to enhance and interpret the gene-disease association studies (Genome Medicine, 2024) [[paper]](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-024-01330-7) [[Code]](https://github.com/bionlp-hzau/PheSeq)
- Heterogeneous biomedical entity representation learning for gene–disease association prediction (Briefings in Bioinformatics, 2024) [[paper]](https://academic.oup.com/bib/article/25/5/bbae380/7735275?login=false) [[Code]](https://github.com/ZhaohanM/FusionGDA)
- MGREL: A multi-graph representation learning-based ensemble learning method for gene-disease association prediction (Computers in Biology and Medicine, 2023) [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0010482523001075?via%3Dihub) [[Code]](https://github.com/ziyang-W/MGREL)
- Multi-domain knowledge graph embeddings for gene-disease association prediction (Journal of Biomedical Semantics, 2023) [[paper]](https://jbiomedsem.biomedcentral.com/articles/10.1186/s13326-023-00291-x) [[Code]](https://github.com/liseda-lab/KGE_Predictions_GD)
- Gene prediction of aging-related diseases based on DNN and Mashup (BMC Bioinformatics, 2021) [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04518-5) [[Code]](https://github.com/jhuaye/GenesRelateDiseases)
- Predicting gene-disease associations from the heterogeneous network using graph embedding (BIBM, 2019) [[paper]](https://ieeexplore.ieee.org/abstract/document/8983134?casa_token=DeIFF4LJjRkAAAAA:nQFsZGhaXQa-rben8QUnDzfBZqnxbqlhgrMyBQp10itqzyFAQ6WKrxAgiTFvOVuN4AFyRSPVvpue)
- Enhancing the prediction of disease–gene associations with multimodal deep learning (Bioinformatics, 2019) [[paper]](https://academic.oup.com/bioinformatics/article/35/19/3735/5368487) [[Code]](https://github.com/luoping1004/dgMDL)
- Large-Scale Discovery of Disease-Disease and Disease-Gene Associations (Scientific Reports, 2016) [[paper]](https://www.nature.com/articles/srep32404)
- Biomedical knowledge graph embeddings for personalized medicine: Predicting disease-gene associations (Expert Systems, 2022) [[paper]](https://onlinelibrary.wiley.com/doi/full/10.1111/exsy.13181)
- KDGene: knowledge graph completion for disease gene prediction using interactional tensor decomposition (Briefings in Bioinformatics, 2024) [[paper]](https://academic.oup.com/bib/article/25/3/bbae161/7644136?searchresult=1) [[Code]](https://github.com/2020MEAI/KDGene)

### Graph neural networks
- Ensemble learning methods and heterogeneous graph network fusion: building drug-gene-disease triple association prediction models (Briefings in Bioinformatics, 2025) [[paper]](https://academic.oup.com/bib/article/26/4/bbaf369/8211399?login=false)
- End-to-end interpretable disease-gene association prediction (Briefings in Bioinformatics, 2023) [[paper]](https://academic.oup.com/bib/article/24/3/bbad118/7091476?login=false) [[Code]](https://github.com/catly/DGP-PGTN)
- Identifying candidate gene–disease associations via graph neural networks (Entropy, 2023) [[paper]](https://www.mdpi.com/1099-4300/25/6/909) [[Code]](https://github.com/pietrocinaglia/gnn_gda)
- Predicting gene-disease associations via graph embedding and graph convolutional networks (BIBM, 2019) [[paper]](https://ieeexplore.ieee.org/abstract/document/8983350)
- VGE: Gene-Disease Association by Variational Graph Embedding (International Journal of Crowd Science, 2024) [[paper]](https://ieeexplore.ieee.org/abstract/document/10530642)
- Disease gene prediction with privileged information and heteroscedastic dropout (Bioinformatics, 2021) [[paper]](https://academic.oup.com/bioinformatics/article/37/Supplement_1/i410/6319695?searchresult=1) [[Code]](https://github.com/juanshu30/Disease-Gene-Prioritization-with-Privileged-Information-and-Heteroscedastic-Dropout)
- A comprehensive graph neural network method for predicting triplet motifs in disease–drug–gene interactions (Bioinformatics, 2025) [[paper]](https://academic.oup.com/bioinformatics/article/41/2/btaf023/7964716?searchresult=1) [[Code]](https://github.com/zhanglabNKU/TriMoGCL)
- Ensemble learning methods and heterogeneous graph network fusion: building drug-gene-disease triple association prediction models (Briefings in Bioinformatics, 2025) [[paper]](https://academic.oup.com/bib/article/26/4/bbaf369/8211399?searchresult=1)

### Large language models
- GPAD: a natural language processing-based application to extract the gene-disease association discovery information from OMIM (BMC BIOINFORMATICS, 2024) [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-024-05693-x)
- Multi-ontology embeddings approach on human-aligned multi-ontologies representation for gene-disease associations prediction (Heliyon, 2023) [[paper]](https://www.cell.com/heliyon/fulltext/S2405-8440(23)08710-8?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2405844023087108%3Fshowall%3Dtrue) [[Code]](https://github.com/Yihao21/MultiOE-4-GDA-Prediction)
- A large language model framework for literature-based disease–gene association prediction (Briefings in Bioinformatics, 2025) [[paper]](https://academic.oup.com/bib/article/26/1/bbaf070/8042066?searchresult=1) [[Code]](https://github.com/ailabstw/LORE)

## Advances and Applications
### Evaluation protocols and Results
* Tele-Knowledge Pre-training for Fault Analysis (ICDE, 2023) [[paper]](https://arxiv.org/abs/2210.11298)
* Pre-training language model incorporating domain-specific heterogeneous knowledge into a unified representation (Expert Systems with Applications, 2023) [[paper]](https://www.sciencedirect.com/science/article/pii/S0957417422023879)
* Deep Bidirectional Language-Knowledge Graph
Pretraining (NIPS, 2022) [[paper]](https://arxiv.org/abs/2210.09338)
* KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation (TACL, 2021) [[paper]](https://aclanthology.org/2021.tacl-1.11.pdf)
* JointGT: Graph-Text Joint Representation Learning for Text Generation from Knowledge Graphs (ACL 2021) [[paper]](https://aclanthology.org/2021.findings-acl.223/)

### Applications
* In silico prediction of novel therapeutic targets using gene--disease association data (Journal of translational medicine, 2017) [[paper]](https://link.springer.com/article/10.1186/s12967-017-1285-6)
* AITeQ: a machine learning framework for Alzheimer’s prediction using a distinctive five-gene signature (Briefings in Bioinformatics, 2024) [[paper]](https://academic.oup.com/bib/article/25/4/bbae291/7692307?searchresult=1) [[Code]](https://github.com/ishtiaque-ahammad/AITeQ)
* Improving drug response prediction via integrating gene relationships with deep learning (Briefings in Bioinformatics, 2024) [[paper]](https://academic.oup.com/bib/article/25/3/bbae153/7642699?searchresult=1#supplementary-data) [[Code]](https://github.com/user15632/DIPK)

## Challenges and Future Directions
* RecInDial: A Unified Framework for Conversational Recommendation with Pretrained Language Models (Arxiv, 2023) [[paper]](https://arxiv.org/pdf/2110.07477.pdf)
* Tele-Knowledge Pre-training for Fault Analysis (ICDE, 2023) [[paper]](https://arxiv.org/abs/2210.11298)
