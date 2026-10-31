# Data

16 classification datasets from OpenML: the 18 classification datasets with at
most 5,000 rows in [TabArena](https://tabarena.ai) (OpenML Study 457), minus
Bioresponse and hiva_agnostic (more than 1,600 features). Twelve are loaded from
TabArena's re-uploads; four (blood-transfusion, diabetes, credit-g,
maternal-health-risk) from their original OpenML ids.

| Dataset | OpenML | Rows | Features | Classes |
|---|---|---:|---:|---:|
| blood-transfusion | 1464 | 748 | 4 | 2 |
| diabetes (Pima) | 37 | 768 | 8 | 2 |
| anneal | 46906 | 898 | 38 | 5 |
| credit-g | 31 | 1,000 | 20 | 2 |
| maternal-health-risk | 46600 | 1,014 | 6 | 3 |
| qsar-biodeg | 46952 | 1,054 | 41 | 2 |
| website_phishing | 46963 | 1,353 | 9 | 3 |
| Fitness_Club | 46927 | 1,500 | 6 | 2 |
| MIC | 46980 | 1,699 | 111 | 8 |
| Is-this-a-good-customer | 46938 | 1,723 | 13 | 2 |
| Marketing_Campaign | 46940 | 2,240 | 25 | 2 |
| hazelnut-spread-contaminant-detection | 46930 | 2,400 | 30 | 2 |
| seismic-bumps | 46956 | 2,584 | 15 | 2 |
| splice | 46958 | 3,190 | 60 | 3 |
| students_dropout_and_academic_success | 46960 | 4,424 | 36 | 3 |
| churn | 46915 | 5,000 | 19 | 2 |

- The datasets are **not included**. `bench_datasets.py` fetches them from OpenML.
- Modifications: the splits; stripping literal quote characters from the
  maternal-health-risk labels (OpenML 46600); converting the one date column
  (Marketing_Campaign's `Dt_Customer`) to days since 1970-01-01; treating the other
  plain text columns as categorical.
- A synthetic dataset (`make_classification`) checks that the pipeline runs
  without network access. It is not part of any comparison.

## Attribution

Licenses as listed on OpenML, or on the UCI page where OpenML gives none
(blood-transfusion, credit-g, maternal-health-risk).

- **blood-transfusion** (CC BY 4.0): Yeh, I-C. (2008). *Blood Transfusion Service Center*. UCI Machine Learning Repository. https://doi.org/10.24432/C5GS39
- **diabetes / Pima Indians Diabetes** (OpenML: "Public"; original owner: National Institute of Diabetes and Digestive and Kidney Diseases): Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/
- **credit-g** (CC BY 4.0): Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- **maternal-health-risk** (CC BY 4.0): Ahmed, M. (2020). *Maternal Health Risk*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DP5D
- **anneal** (CC BY 4.0): *Annealing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5RW2F
- **qsar-biodeg** (CC BY 4.0): Mansouri, K., et al. (2013). Quantitative structure–activity relationship models for ready biodegradability of chemicals. *Journal of Chemical Information and Modeling*, 53(4), 867–878. https://doi.org/10.1021/ci4000213
- **website_phishing** (CC BY 4.0): Abdelhamid, N., Ayesh, A., & Thabtah, F. (2014). Phishing detection based associative classification data mining. *Expert Systems with Applications*, 41(13), 5948–5959. https://doi.org/10.1016/j.eswa.2014.03.019
- **Fitness_Club** (Public Domain): Kaggle user Ddosad (2023). *Fitness Club Dataset for ML Classification*. https://www.kaggle.com/datasets/ddosad/datacamps-data-science-associate-certification
- **MIC** (CC BY 4.0): Golovenkin, S. E., et al. (2020). Trajectories, bifurcations, and pseudo-time in large clinical datasets: applications to myocardial infarction and diabetes data. *GigaScience*, 9(11), giaa128. https://doi.org/10.1093/gigascience/giaa128
- **Is-this-a-good-customer** (CC0): Kaggle user Podsyp (2020). *Is This a Good Customer?* https://www.kaggle.com/datasets/podsyp/is-this-a-good-customer
- **Marketing_Campaign** (OpenML: "Public"): Saldanha, R. (2020). *Marketing Campaign*. Kaggle. https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign
- **hazelnut-spread-contaminant-detection** (CC BY-SA): Ricci, M., et al. (2021). Machine-learning-based microwave sensing: A case study for the food industry. *IEEE Journal on Emerging and Selected Topics in Circuits and Systems*, 11(3), 503–514. https://doi.org/10.1109/JETCAS.2021.3097699 ; Urbinati, L., et al. (2020). A machine-learning based microwave sensing approach to food contaminant detection. *2020 IEEE ISCAS*. https://doi.org/10.1109/ISCAS45731.2020.9181293
- **seismic-bumps** (CC BY 4.0): Sikora, M., & Wróbel, Ł. (2010). Application of rule induction algorithms for analysis of data collected by seismic hazard monitoring systems in coal mines. *Archives of Mining Sciences*, 55(1), 91–114.
- **splice** (CC BY 4.0): Towell, G. G., & Shavlik, J. W. (1994). Knowledge-based artificial neural networks. *Artificial Intelligence*, 70(1–2), 119–165. https://doi.org/10.1016/0004-3702(94)90105-8
- **students_dropout_and_academic_success** (CC BY 4.0): Martins, M. V., et al. (2021). Early prediction of student's performance in higher education: A case study. *Trends and Applications in Information Systems and Technologies*, Volume 1. Springer. https://doi.org/10.1007/978-3-030-72657-7_16
- **churn** (OpenML: MIT License): The original source is lost; OpenML points to PMLB (https://github.com/EpistasisLab/pmlb/tree/master/datasets/churn) and cites a book review: Marcoulides, G. A. (2005). Review of *Discovering Knowledge in Data: An Introduction to Data Mining*. *Journal of the American Statistical Association*, 100(472), 1465. https://doi.org/10.1198/jasa.2005.s61
