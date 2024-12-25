

This program starts by reading the AHS data obtained during the data preparation phase. The main functionality is implemented in the `main_preprocessing` and `utils` Python files. The following steps are performed:

1. **Categorical Columns Transformation**:  
   - Categorical columns, particularly those related to healthcare utilization (e.g., ER visits), are converted into dummy variables to facilitate analysis.

2. **Merging Duplicate Rows**:  
   - Rows with the same values for the same date were identified as duplicates. These rows were merged to ensure only unique entries for each date.

3. **Processing the `sex` Column**:  
   - The `sex` column was filtered to keep only the records for males, and boolean values (True/False) were replaced with numeric values (1/0).

4. **Handling Missing Visit Records**:  
   - Some files were incomplete and missing certain visit records. These missing visits were added to the dataset with a value of zero.

5. **Truncating Homeless Outcome Records**:  
   - For individuals with multiple homelessness outcomes, only the first recorded outcome was retained. All subsequent records were truncated.

6. **Two-Year Observation Window Aggregation**:  
   - Data was aggregated into a two-year observation window. If an AMH (Adult Mental Health) diagnosis occurred, it served as the trigger for the aggregation period. This step ensures that states are built based on either an AMH diagnosis or a two-year observation window.

7. **Renaming Columns for Ease of Access**:  
   - Columns were renamed using prefixes for clarity and organization:
     - `m:` for metadata.
     - `o:` for observation features.
     - `a:` for actions.
     - `r:` for rewards.
   - This naming convention facilitates easier access to features for building episodes.

8. **Train-Test-Validation Split**:  
   - The dataset was split into training, testing, and validation subsets using a 70-20-10 ratio.

These steps prepare the dataset for further analysis, ensuring data quality, consistency, and ease of use for modeling and research.


Next step I will work on the actions to make them encoded and work on the reward function to have one columns as the reward before spliting and saving
