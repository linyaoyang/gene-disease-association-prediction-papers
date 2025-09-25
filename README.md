# Gene-disease-association-prediction-papers
A collection of papers and resources about gene disease association prediction.

## Table of Contents
- [Gene-disease-association-prediction-papers](#Gene-disease-association-prediction-papers)
  - [Related Surveys](#related-surveys)
  - [Datasets](#Datasets)
    - [Gene-related datasets](#gene-related-datasets)
    - [Disease-related datasets](#disease-related-datasets)
    - [GDA datasets](#gda-datasets)
    - [Extended resources](#extended-resources)
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
* Recent advances in predicting gene–disease associations (F1000Research, 2017) [[paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC5414807/)
* Recent advances in network-based methods for disease gene prediction (Briefings in bioinformatics, 2021) [[paper]](https://academic.oup.com/bib/article/22/4/bbaa303/6023077?login=false)

## Datasets
### Gene-related datasets
- **HumanNet** [[paper]](https://pubmed.ncbi.nlm.nih.gov/34747468/) [[Link]](https://www.inetbio.org/humannet/)
- **GWAS** [[paper]](https://pubmed.ncbi.nlm.nih.gov/36350656/) [[Link]](https://www.ebi.ac.uk/gwas/)
- **GO** [[paper]](https://pubmed.ncbi.nlm.nih.gov/30395331/) [[Link]](https://geneontology.org/)
- **BioGPS** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/gb-2009-10-11-r130) [[Link]](https://biogps.org/dataset/)

### Disease-related datasets
- **MeSH** [[paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC35238/) [[Link]](https://www.nlm.nih.gov/mesh/meshhome.html)
- **HPO** [[paper]](https://pubmed.ncbi.nlm.nih.gov/33264411/) [[Link]](https://hpo.jax.org/)
- **DO** [[paper]](https://pubmed.ncbi.nlm.nih.gov/22080554/) [[Link]](https://disease-ontology.org/)

### GDA datasets
- **CTD** [[paper]](https://academic.oup.com/nar/article/53/D1/D1328/7816860) [[Link]](https://ctdbase.org/)
- **OMIM** [[paper]](https://pubmed.ncbi.nlm.nih.gov/15608251/) [[Link]](https://www.omim.org/)
- **DisGeNET** [[paper]](https://pubmed.ncbi.nlm.nih.gov/27924018/) [[Link]](https://disease-ontology.org/)

### Extended resources
- **HPRD** [[paper]](https://pubmed.ncbi.nlm.nih.gov/18988627/) [[Link]](http://www.hprd.org/)
- **BIND** [[paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC165503/) [[Link]](http://bind.ca/)
- **BioGRID** [[paper]](https://pubmed.ncbi.nlm.nih.gov/33070389/) [[Link]](https://thebiogrid.org/)
- **IntAct** [[paper]](https://pubmed.ncbi.nlm.nih.gov/22121220/) [[Link]](https://www.ebi.ac.uk/intact/)
- **STRING** [[paper]](https://academic.oup.com/nar/article/49/D1/D605/6006194?login=false) [[Link]](https://string-db.org/)
- **KEGG** [[paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC102409/) [[Link]](https://www.genome.jp/kegg/)
- **Reactome** [[paper]](https://pubmed.ncbi.nlm.nih.gov/31691815/) [[Link]](https://reactome.org/)
- **DrugBank** [[paper]](https://pubmed.ncbi.nlm.nih.gov/37953279/) [[Link]](https://go.drugbank.com/)
- **ChEMBL** [[paper]](https://pubmed.ncbi.nlm.nih.gov/21948594/) [[Link]](https://www.ebi.ac.uk/chembl/)

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
- Machine learning approaches for the discovery of gene–gene interactions in disease data (Briefings in Bioinformatics, 2013) [[paper]](https://academic.oup.com/bib/article/14/2/251/208670?login=false)

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
- BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model (Arxiv, 2025) [[paper]](https://arxiv.org/abs/2505.23579) [[Code]](https://github.com/bowang-lab/BioReason)

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
* Comparative study of joint analysis of microarray gene expression data in survival prediction and risk assessment of breast cancer patients (Briefings in Bioinformatics, 2016) [[paper]](https://academic.oup.com/bib/article/17/5/771/2262453?searchresult=1)

## Challenges and Future Directions
* RecInDial: A Unified Framework for Conversational Recommendation with Pretrained Language Models (Arxiv, 2023) [[paper]](https://arxiv.org/pdf/2110.07477.pdf)
* Tele-Knowledge Pre-training for Fault Analysis (ICDE, 2023) [[paper]](https://arxiv.org/abs/2210.11298)
