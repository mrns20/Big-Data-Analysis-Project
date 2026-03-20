Big Data Analysis Project — Billionaires Statistics Dataset
 English | Ελληνικά


<a name="english"></a>
 English
Overview
This project performs a comprehensive big data analysis on the Billionaires Statistics Dataset, applying and comparing three clustering algorithms: Hierarchical Clustering, K-Means, and DBSCAN. The goal is to identify patterns and groupings among billionaires based on attributes such as net worth, age, country, industry, and macroeconomic indicators.

Dataset
The dataset contains statistics about billionaires worldwide, including:

rank — Wealth ranking
finalWorth — Net worth in USD
category — Industry/business category
personName — Full name
age — Age of the billionaire
country — Country of residence
source — Source of wealth
selfMade — Whether the billionaire is self-made
gender — Gender
cpi_country — Consumer Price Index of their country
gdp_country — GDP of their country
gross_tertiary_education_enrollment — Tertiary education enrollment rate
total_tax_rate_country — Total tax rate in their country

Methodology
1. Data Preprocessing

Loaded and inspected the dataset using pandas
Checked for missing values and duplicates
Encoded categorical and boolean columns (category, selfMade, gender) using LabelEncoder
Removed records with missing values in key columns
Fixed ranking inconsistencies after record removal
Reduced features from 35 to 13 through correlation analysis (Pearson)

2. Outlier Detection

Visualized potential outliers using box plots for key numerical columns (finalWorth, age, gdp_country, total_tax_rate_country)

3. Normalization

Applied Z-score normalization (StandardScaler) to handle outliers and ensure fair comparison across features

4. Clustering Algorithms
AlgorithmKey ParametersDistance MetricHierarchical Clusteringnum_of_clusters=3Cityblock (Manhattan)K-Meansk=3, evaluated for k=3–10EuclideanDBSCANeps=0.3, min_samples=15Euclidean

6. Evaluation Metrics

Silhouette Coefficient — measures clustering quality (range: -1 to 1)
SSE (Sum of Squared Errors) — for K-Means optimization
Execution time — for algorithm comparison


Analysis Examples
Two feature pair combinations were analyzed:

Example 1: finalWorth vs age
Example 2: finalWorth vs total_tax_rate_country


Requirements
bashpip install pandas matplotlib seaborn scikit-learn scipy
Python version: 3.8+

How to Run
Clone the repository:

bash   git clone https://github.com/mrns20/Big-Data-Analysis-Project.git
   cd Big-Data-Analysis-Project

Install dependencies:

bash   pip install pandas matplotlib seaborn scikit-learn scipy

Run the analysis script:

bash   python Dataset_analysis_code.py

Note: Make sure the Billionaires Statistics Dataset.csv file is in the same directory as the script.


Output
The script generates:

Preprocessed and normalized CSV files
Dendrograms (Hierarchical Clustering)
Scatter plots for each clustering method
SSE and Silhouette Coefficient plots for K-Means (k=3 to 10)
Console output comparing all three algorithms' performance


-------------------------------------------------------------

<a name="greek"></a>
Ελληνικά
Επισκόπηση
Αυτό το project πραγματοποιεί ολοκληρωμένη ανάλυση μεγάλων δεδομένων στο Billionaires Statistics Dataset, εφαρμόζοντας και συγκρίνοντας τρεις αλγορίθμους συσταδοποίησης: Ιεραρχική Συσταδοποίηση, K-Means και DBSCAN. Στόχος είναι η ανίχνευση μοτίβων και ομαδοποιήσεων μεταξύ δισεκατομμυριούχων βάσει χαρακτηριστικών όπως ο πλούτος, η ηλικία, η χώρα, ο κλάδος δραστηριότητας και μακροοικονομικοί δείκτες.

 Σύνολο Δεδομένων
Το dataset περιέχει στατιστικά στοιχεία για δισεκατομμυριούχους παγκοσμίως, συμπεριλαμβανομένων:

rank — Κατάταξη βάσει πλούτου
finalWorth — Καθαρή περιουσία σε USD
category — Κατηγορία κλάδου/επιχείρησης
personName — Πλήρες όνομα
age — Ηλικία
country — Χώρα διαμονής
source — Πηγή πλούτου
selfMade — Αν ο δισεκατομμυριούχος αυτοδημιουργήθηκε
gender — Φύλο
cpi_country — Δείκτης Τιμών Καταναλωτή χώρας
gdp_country — ΑΕΠ χώρας
gross_tertiary_education_enrollment — Ποσοστό εγγραφής στην τριτοβάθμια εκπαίδευση
total_tax_rate_country — Συνολικός φορολογικός συντελεστής χώρας


 Μεθοδολογία
1. Προεπεξεργασία Δεδομένων

Φόρτωση και επιθεώρηση δεδομένων με pandas
Έλεγχος για ελλιπείς τιμές και διπλότυπα
Κωδικοποίηση κατηγορικών και boolean στηλών (category, selfMade, gender) με LabelEncoder
Αφαίρεση εγγραφών με ελλιπείς τιμές σε βασικές στήλες
Διόρθωση κατάταξης μετά την αφαίρεση εγγραφών
Μείωση χαρακτηριστικών από 35 σε 13 μέσω ανάλυσης συσχέτισης (Pearson)

2. Ανίχνευση Ακραίων Τιμών

Οπτικοποίηση ακραίων τιμών με box plots για βασικές αριθμητικές στήλες

3. Κανονικοποίηση

Εφαρμογή κανονικοποίησης Z-score (StandardScaler) για αντιμετώπιση ακραίων τιμών και δίκαιη σύγκριση χαρακτηριστικών

4. Αλγόριθμοι Συσταδοποίησης
ΑλγόριθμοςΒασικές ΠαράμετροιΜετρική ΑπόστασηςΙεραρχική Συσταδοποίησηnum_of_clusters=3Cityblock (Μανχάταν)K-Meansk=3, αξιολόγηση για k=3–10ΕυκλείδειαDBSCANeps=0.3, min_samples=15Ευκλείδεια
5. Μετρικές Αξιολόγησης

Silhouette Coefficient — μέτρο ποιότητας συσταδοποίησης (εύρος: -1 έως 1)
SSE (Συνολικό Τετραγωνικό Σφάλμα) — για βελτιστοποίηση K-Means
Χρόνος εκτέλεσης — για σύγκριση αλγορίθμων


🔬 Παραδείγματα Ανάλυσης
Αναλύθηκαν δύο ζεύγη χαρακτηριστικών:

Παράδειγμα 1: finalWorth vs age
Παράδειγμα 2: finalWorth vs total_tax_rate_country


 Απαιτήσεις
bashpip install pandas matplotlib seaborn scikit-learn scipy
Έκδοση Python: 3.8+

 Εκτέλεση

Κλωνοποίηση αποθετηρίου:

bash   git clone https://github.com/mrns20/Big-Data-Analysis-Project.git
   cd Big-Data-Analysis-Project

Εγκατάσταση εξαρτήσεων:

bash   pip install pandas matplotlib seaborn scikit-learn scipy

Εκτέλεση σεναρίου ανάλυσης:

bash   python Dataset_analysis_code.py

Σημείωση: Βεβαιωθείτε ότι το αρχείο Billionaires Statistics Dataset.csv βρίσκεται στον ίδιο φάκελο με το script.


 Αποτελέσματα
Το script παράγει:

Επεξεργασμένα και κανονικοποιημένα αρχεία CSV
Δεντρογράμματα (Ιεραρχική Συσταδοποίηση)
Διαγράμματα διασποράς για κάθε μέθοδο συσταδοποίησης
Γραφικές παραστάσεις SSE και Silhouette Coefficient για K-Means (k=3 έως 10)
Σύγκριση απόδοσης και των τριών αλγορίθμων στην κονσόλα
