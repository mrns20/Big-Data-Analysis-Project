# Big Data Analysis Project — Billionaires Statistics Dataset

> 🇬🇧 [English](#english) | 🇬🇷 [Ελληνικά](#greek)

---

<a name="english"></a>
## 🇬🇧 English

### Overview

This project performs a comprehensive big data analysis on the **Billionaires Statistics Dataset**, applying and comparing three clustering algorithms: **Hierarchical Clustering**, **K-Means**, and **DBSCAN**. The goal is to identify patterns and groupings among billionaires based on attributes such as net worth, age, country, industry, and macroeconomic indicators.

---

### 📁 Repository Structure

```
Big-Data-Analysis-Project/
│
├── Billionaires Statistics Dataset.csv     # Original dataset
├── Dataset_analysis_code.py               # Main Python analysis script
├── WrittenReport_inGreek.pdf              # Written report (in Greek)
├── Εργασία-ΔΧΜΚ-Απρ24.pdf               # Assignment description (in Greek)
└── README.md
```

---

### 📊 Dataset

The dataset contains statistics about billionaires worldwide. Each record includes the billionaire's wealth ranking (`rank`) and net worth in USD (`finalWorth`), along with personal attributes such as full name, age, country of residence, gender, and whether they are self-made. It also captures the industry or business category they operate in and the primary source of their wealth. On the macroeconomic side, each record includes the Consumer Price Index (`cpi_country`), GDP (`gdp_country`), tertiary education enrollment rate, and total tax rate of the billionaire's country of residence.

---

### ⚙️ Methodology

#### Data Preprocessing

The raw dataset was loaded and inspected using **pandas**. After checking for missing values and duplicates, categorical and boolean columns — specifically `category`, `selfMade`, and `gender` — were encoded into numerical form using **LabelEncoder**. Records with missing values in key columns such as `age`, `country`, `cpi_country`, `gdp_country`, `gross_tertiary_education_enrollment`, and `total_tax_rate_country` were removed, reducing the dataset from 2,641 to 2,408 records. The `rank` column was then recalculated to reflect the updated dataset. A **Pearson correlation analysis** was subsequently performed to identify and remove redundant features, reducing the total number of columns from 35 to 13.

#### Outlier Detection

Potential outliers were visualized using **box plots** for the key numerical columns `finalWorth`, `age`, `gdp_country`, and `total_tax_rate_country`, providing an overview of the data distribution and the presence of extreme values.

#### Normalization

**Z-score normalization** was applied via scikit-learn's `StandardScaler` to all numerical features used in clustering. This method was chosen over min-max normalization because it handles outliers more robustly, ensuring that no single feature dominates the distance calculations during clustering.

#### Clustering Algorithms

Three clustering algorithms were implemented and compared. **Hierarchical Clustering** was applied using the Ward linkage method with a Cityblock (Manhattan) distance metric, and the results were visualized through a dendrogram. **K-Means** was run with Euclidean distance (the only metric supported in Python's implementation), and evaluated across different values of k ranging from 3 to 10 using both SSE and the Silhouette Coefficient. **DBSCAN** was applied with `eps=0.3` and `min_samples=15`, automatically determining the number of clusters and labeling low-density points as noise.

#### Evaluation

Each algorithm was assessed using the **Silhouette Coefficient** (ranging from -1 to 1, where higher values indicate better-defined clusters), **SSE** for K-Means optimization, and **execution time** for a direct performance comparison between methods.

---

### 🔬 Analysis Examples

Two pairs of features were used to demonstrate and compare the clustering algorithms. The first example analyzed `finalWorth` against `age`, while the second examined `finalWorth` against `total_tax_rate_country`.

---

### 🛠️ Requirements

```bash
pip install pandas matplotlib seaborn scikit-learn scipy
```

**Python version:** 3.8+

---

### ▶️ How to Run

Clone the repository and navigate into it with `git clone https://github.com/mrns20/Big-Data-Analysis-Project.git && cd Big-Data-Analysis-Project`. Then install the required dependencies using `pip install pandas matplotlib seaborn scikit-learn scipy`, and finally run the analysis script with `python Dataset_analysis_code.py`. Make sure the `Billionaires Statistics Dataset.csv` file is located in the same directory as the script before running.

---

### 📈 Output

Running the script produces a series of preprocessed and normalized CSV files, as well as a set of visualizations including dendrograms for hierarchical clustering, scatter plots for each of the three clustering methods, and SSE and Silhouette Coefficient plots for K-Means across different values of k. A summary comparing the Silhouette Coefficient and execution time of all three algorithms is also printed to the console.

---

### 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [GeeksForGeeks — Outlier Detection in Python](https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/)
- [Matplotlib Subplots](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)
- [Seaborn Tutorial](https://www.geeksforgeeks.org/python-seaborn-tutorial/)
- [StandardScaler — Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

<a name="greek"></a>
## 🇬🇷 Ελληνικά

### Επισκόπηση

Αυτό το project πραγματοποιεί ολοκληρωμένη ανάλυση μεγάλων δεδομένων στο **Billionaires Statistics Dataset**, εφαρμόζοντας και συγκρίνοντας τρεις αλγορίθμους συσταδοποίησης: **Ιεραρχική Συσταδοποίηση**, **K-Means** και **DBSCAN**. Στόχος είναι η ανίχνευση μοτίβων και ομαδοποιήσεων μεταξύ δισεκατομμυριούχων βάσει χαρακτηριστικών όπως ο πλούτος, η ηλικία, η χώρα, ο κλάδος δραστηριότητας και μακροοικονομικοί δείκτες.

---

### 📁 Δομή Αποθετηρίου

```
Big-Data-Analysis-Project/
│
├── Billionaires Statistics Dataset.csv     # Αρχικό σύνολο δεδομένων
├── Dataset_analysis_code.py               # Κύριο αρχείο Python ανάλυσης
├── WrittenReport_inGreek.pdf              # Γραπτή αναφορά (στα Ελληνικά)
├── Εργασία-ΔΧΜΚ-Απρ24.pdf               # Εκφώνηση εργασίας (στα Ελληνικά)
└── README.md
```

---

### 📊 Σύνολο Δεδομένων

Το dataset περιέχει στατιστικά στοιχεία για δισεκατομμυριούχους παγκοσμίως. Κάθε εγγραφή περιλαμβάνει την κατάταξη βάσει πλούτου (`rank`) και την καθαρή περιουσία σε USD (`finalWorth`), καθώς και προσωπικά χαρακτηριστικά όπως το πλήρες όνομα, η ηλικία, η χώρα διαμονής, το φύλο και αν ο δισεκατομμυριούχος αυτοδημιουργήθηκε. Επίσης καταγράφονται ο κλάδος δραστηριότητας και η κύρια πηγή πλούτου. Σε μακροοικονομικό επίπεδο, κάθε εγγραφή συμπεριλαμβάνει τον Δείκτη Τιμών Καταναλωτή (`cpi_country`), το ΑΕΠ (`gdp_country`), το ποσοστό εγγραφής στην τριτοβάθμια εκπαίδευση και τον συνολικό φορολογικό συντελεστή της χώρας διαμονής.

---

### ⚙️ Μεθοδολογία

#### Προεπεξεργασία Δεδομένων

Το αρχικό dataset φορτώθηκε και επιθεωρήθηκε με τη βοήθεια της βιβλιοθήκης **pandas**. Μετά τον έλεγχο για ελλιπείς τιμές και διπλότυπα, οι κατηγορικές και boolean στήλες — συγκεκριμένα οι `category`, `selfMade` και `gender` — κωδικοποιήθηκαν αριθμητικά με τη χρήση **LabelEncoder**. Αφαιρέθηκαν εγγραφές με ελλιπείς τιμές σε βασικές στήλες, μειώνοντας το σύνολο από 2.641 σε 2.408 εγγραφές. Η στήλη `rank` επανυπολογίστηκε για να αντικατοπτρίζει το ενημερωμένο dataset. Ακολούθησε **ανάλυση συσχέτισης Pearson** για την αναγνώριση και αφαίρεση πλεονασματικών χαρακτηριστικών, μειώνοντας τον αριθμό των στηλών από 35 σε 13.

#### Ανίχνευση Ακραίων Τιμών

Οι πιθανές ακραίες τιμές οπτικοποιήθηκαν μέσω **box plots** για τις βασικές αριθμητικές στήλες `finalWorth`, `age`, `gdp_country` και `total_tax_rate_country`, παρέχοντας μια σαφή εικόνα της κατανομής των δεδομένων και της παρουσίας ακραίων τιμών.

#### Κανονικοποίηση

Εφαρμόστηκε **κανονικοποίηση Z-score** μέσω του `StandardScaler` της scikit-learn σε όλα τα αριθμητικά χαρακτηριστικά που χρησιμοποιήθηκαν στη συσταδοποίηση. Η μέθοδος αυτή επιλέχθηκε έναντι της min-max κανονικοποίησης λόγω της καλύτερης αντιμετώπισης ακραίων τιμών, διασφαλίζοντας ότι κανένα χαρακτηριστικό δεν κυριαρχεί στους υπολογισμούς απόστασης κατά τη συσταδοποίηση.

#### Αλγόριθμοι Συσταδοποίησης

Υλοποιήθηκαν και συγκρίθηκαν τρεις αλγόριθμοι συσταδοποίησης. Η **Ιεραρχική Συσταδοποίηση** εφαρμόστηκε με τη μέθοδο Ward και μετρική απόστασης Cityblock (Μανχάταν), με αποτελέσματα που οπτικοποιήθηκαν μέσω δεντρογράμματος. Ο **K-Means** εκτελέστηκε με Ευκλείδεια απόσταση και αξιολογήθηκε για διάφορες τιμές του k από 3 έως 10, χρησιμοποιώντας τόσο το SSE όσο και τον Silhouette Coefficient. Ο **DBSCAN** εφαρμόστηκε με `eps=0.3` και `min_samples=15`, προσδιορίζοντας αυτόματα τον αριθμό των συστάδων και χαρακτηρίζοντας ως θόρυβο τα σημεία χαμηλής πυκνότητας.

#### Αξιολόγηση

Κάθε αλγόριθμος αξιολογήθηκε με βάση τον **Silhouette Coefficient** (εύρος -1 έως 1, όπου υψηλότερες τιμές υποδηλώνουν καλύτερα ορισμένες συστάδες), το **SSE** για βελτιστοποίηση K-Means, και τον **χρόνο εκτέλεσης** για άμεση σύγκριση απόδοσης μεταξύ των μεθόδων.

---

### 🔬 Παραδείγματα Ανάλυσης

Χρησιμοποιήθηκαν δύο ζεύγη χαρακτηριστικών για την επίδειξη και σύγκριση των αλγορίθμων συσταδοποίησης. Το πρώτο παράδειγμα ανέλυσε τη σχέση μεταξύ `finalWorth` και `age`, ενώ το δεύτερο εξέτασε τη σχέση μεταξύ `finalWorth` και `total_tax_rate_country`.

---

### 🛠️ Απαιτήσεις

```bash
pip install pandas matplotlib seaborn scikit-learn scipy
```

**Έκδοση Python:** 3.8+

---

### ▶️ Εκτέλεση

Κλωνοποιήστε το αποθετήριο και μεταβείτε σε αυτό με `git clone https://github.com/mrns20/Big-Data-Analysis-Project.git && cd Big-Data-Analysis-Project`. Στη συνέχεια εγκαταστήστε τις απαιτούμενες βιβλιοθήκες με `pip install pandas matplotlib seaborn scikit-learn scipy` και εκτελέστε το script ανάλυσης με `python Dataset_analysis_code.py`. Βεβαιωθείτε ότι το αρχείο `Billionaires Statistics Dataset.csv` βρίσκεται στον ίδιο φάκελο με το script πριν από την εκτέλεση.

---

### 📈 Αποτελέσματα

Η εκτέλεση του script παράγει μια σειρά από επεξεργασμένα και κανονικοποιημένα αρχεία CSV, καθώς και οπτικοποιήσεις που περιλαμβάνουν δεντρογράμματα για την ιεραρχική συσταδοποίηση, διαγράμματα διασποράς για κάθε μέθοδο και γραφικές παραστάσεις SSE και Silhouette Coefficient για τον K-Means σε διάφορες τιμές k. Επίσης εκτυπώνεται στην κονσόλα συνοπτική σύγκριση του Silhouette Coefficient και του χρόνου εκτέλεσης και των τριών αλγορίθμων.

---

### 📚 Αναφορές

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [GeeksForGeeks — Ανίχνευση Ακραίων Τιμών σε Python](https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/)
- [Matplotlib Subplots](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)
- [Seaborn Tutorial](https://www.geeksforgeeks.org/python-seaborn-tutorial/)
- [StandardScaler — Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
