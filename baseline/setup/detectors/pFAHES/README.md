# pFAHES
python version of FAHES

# FAHES: A Robust Disguised Missing Values Detector
Missing values are common in real-world data and may seriously affect data analytics such as simple statistics and hypothesis testing. Generally speaking, there are two types of missing values:  explicitly missing values (i.e. NULL values), and implicitly missing values (a.k.a. disguised missing values (DMVs)) such as "11111111" for a phone number and "Some college" for education. While detecting explicitly missing values is trivial, detecting DMVs is not; the essential challenge is the lack of standardization about how DMVs are generated. 

FAHES is a system for detecting DMVs from two angles: DMVs as detectable outliers and as detectable inliers.  For DMVs as outliers, we propose a syntactic outlier detection module for categorical data, and a density-based outlier detection module for numerical values. For DMVs as inliers, we propose a method that detects DMVs which follow either missing-completely-at-random or missing-at-random models. 


Get the code

    # clone using https
    git clone https://github.com/qahtanaa/pFAHES.git
    
    cd pFAHES
    
Install dependencies:

    To run FAHES, you need only to install pandas

Run the code
    
    python main.py Path_to_data_files/file_name.csv folder_for_results/ tool_code

tool_code: 1 for detecting syntactical DMVs,
           2 for detecting DMVs that replace missing at random values, 
           3 for detecting numerical DMVs, and
           4 for detecting all types of DMVs
