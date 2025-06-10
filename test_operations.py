import numpy as np
import pandas as pd
import re
import random
from collections import Counter

def process_test_data(test_csv_path, original_train_csv_path="data/train.csv"):
    """
    Apply all training data transformations to test data
    """
    # Load test data and original training data
    test_df = pd.read_csv(test_csv_path)
    train_df = pd.read_csv(original_train_csv_path)
    
    print("Original test data shape:", test_df.shape)
    print("Missing values in test data:")
    print(test_df.isna().sum())
    
    # 1. AGE CLASS PREDICTION
    print("\n1. Processing age classes...")
    
    # Create fare lookup from training data (already computed)
    age_df_train = train_df[train_df["Age"].notna()].copy()
    
    # Create age class for training data
    def create_age_class_column(df):
        df = df.copy()
        df["age_class"] = 0
        df.loc[df["Age"] < 18, "age_class"] = 1
        df.loc[(df["Age"] >= 18) & (df["Age"] < 30), "age_class"] = 2
        df.loc[(df["Age"] >= 30) & (df["Age"] < 50), "age_class"] = 3
        df.loc[df["Age"] >= 50, "age_class"] = 4
        return df
    
    age_df_train = create_age_class_column(age_df_train)
    fare_lookup = age_df_train.groupby(["Pclass", "age_class"])["Fare"].mean().to_dict()
    
    def predict_age_class(row, fare_lookup):
        if pd.notna(row["Age"]):
            if row["Age"] < 18:
                return 1
            elif row["Age"] < 30:
                return 2
            elif row["Age"] < 50:
                return 3
            else:
                return 4
        else:
            pclass = row["Pclass"]
            fare = row["Fare"]
            
            # Handle missing fare
            if pd.isna(fare):
                return 2  # Default to young adult
            
            pclass_fares = {k: v for k, v in fare_lookup.items() if k[0] == pclass}
            
            if not pclass_fares:
                return 2
                
            closest_age_class = min(pclass_fares.keys(), 
                                   key=lambda x: abs(pclass_fares[x] - fare))[1]
            return closest_age_class
    
    test_df["age_class"] = test_df.apply(
        lambda row: predict_age_class(row, fare_lookup), axis=1
    )
    
    # 2. FILL MISSING AGES
    print("2. Filling missing ages...")
    
    # Create age lookup from training data
    train_with_age_class = train_df[train_df["Age"].notna()].copy()
    train_with_age_class["age_class"] = train_with_age_class.apply(
        lambda row: predict_age_class(row, fare_lookup), axis=1
    )
    
    age_lookup_df = train_with_age_class.groupby(["Pclass", "age_class"])["Age"].mean().reset_index()
    age_lookup_df.columns = ["Pclass", "age_class", "predicted_age"]
    
    test_df = test_df.merge(age_lookup_df, on=["Pclass", "age_class"], how="left")
    test_df["Age"] = test_df["Age"].fillna(test_df["predicted_age"])
    test_df = test_df.drop("predicted_age", axis=1)
    
    # 3. FAMILY HIERARCHY
    print("3. Processing family hierarchy...")
    
    test_df["lastname"] = test_df["Name"].str.split(",").str[0]
    
    def check_family_hierarchy_test(row, full_df):
        dependent_counter = 0
        dependent_threshold_lower = 18
        dependent_threshold_upper = 45
        
        # Use the test dataframe itself for family analysis
        temp_dependent = full_df[(full_df["Name"].str.contains(row["lastname"], na=False)) & 
                                (full_df["PassengerId"] != row["PassengerId"])]
        temp_dependent_age = list(temp_dependent["Age"])
        
        for age in temp_dependent_age:
            if age < dependent_threshold_lower or age > dependent_threshold_upper:
                dependent_counter += 1
        
        adult_counter = len(temp_dependent_age) - dependent_counter
        
        if row["Age"] < dependent_threshold_lower or row["Age"] > dependent_threshold_upper:
            return 0, adult_counter
        
        return dependent_counter, adult_counter
    
    result = test_df.apply(lambda row: check_family_hierarchy_test(row, test_df), axis=1)
    test_df["dependent_count"], test_df["adult_count"] = zip(*result)
    
    # 4. DROP NAME COLUMNS
    print("4. Dropping name columns...")
    test_df = test_df.drop(["Name", "lastname"], axis=1)
    
    # 5. GENDER TO NUMERIC
    print("5. Converting gender to numeric...")
    test_df["Sex"] = test_df["Sex"].map({'male': 1, "female": 0})
    
    # 6. EMBARKED TO NUMERIC
    print("6. Processing embarked column...")
    test_df["Embarked"] = test_df["Embarked"].map({"S": 3, "Q": 1, "C": 2})
    
    # Fill missing embarked values (use mode from training or default to 3 like in training)
    test_df["Embarked"] = test_df["Embarked"].fillna(3)
    
    # 7. FAMILY MEMBERS
    print("7. Creating family members column...")
    test_df["family_members"] = test_df["SibSp"] + test_df["Parch"]
    
    # 8. TICKET PROCESSING
    print("8. Processing tickets...")
    
    def get_ticket_analysis(row):
        ticket = row["Ticket"]
        if len(ticket.split(" ")) > 1:
            ticket_prefix = ticket.split(" ")[0]
            ticket_no = ticket.split(" ")[1]
        else:
            ticket_prefix = ""
            ticket_no = ticket
        return ticket_prefix, ticket_no
    
    results = test_df.apply(get_ticket_analysis, axis=1)
    test_df["ticket_prefix"], test_df["ticket_no"] = zip(*results)
    
    # Shared tickets analysis
    share_lookup = test_df["ticket_no"].value_counts().to_dict()
    shared_tickets_list = [k for k, v in share_lookup.items() if v > 1]
    
    def is_shared(row, shared_tickets_list):
        if row["ticket_no"] in shared_tickets_list:
            return 1
        return 0
    
    test_df["is_shared"] = test_df.apply(
        lambda row: is_shared(row, shared_tickets_list), axis=1
    )
    
    # Convert ticket prefix to numeric using consistent mapping
    # Create mapping from all unique prefixes in test data
    all_prefixes = list(test_df["ticket_prefix"].unique())
    prefix_mapping = {prefix: i for i, prefix in enumerate(sorted(all_prefixes))}
    test_df["ticket_prefix"] = test_df["ticket_prefix"].map(prefix_mapping).fillna(-1)
    
    test_df = test_df.drop(["Ticket"], axis=1)
    
    # Total occupants
    cabin_share_count = test_df["ticket_no"].value_counts().to_dict()
    test_df["total_occupants"] = test_df["ticket_no"].map(cabin_share_count)
    
    # Adult share count
    def cabin_shared_with_adult(row, full_df):
        if row["is_shared"] == 1:
            ticket_no = row["ticket_no"]
            shared_passengers = full_df[full_df["ticket_no"] == ticket_no]
            shared_passengers_age = shared_passengers["Age"].to_list()
            return sum(1 for age in shared_passengers_age if age >= 18)
        return 0
    
    test_df["adult_share_count"] = test_df.apply(
        lambda row: cabin_shared_with_adult(row, test_df), axis=1
    )
    
    # 9. CABIN PROCESSING
    print("9. Processing cabins...")
    
    def process_cabin(c):
        if pd.isna(c):
            return None, None
        
        match = re.match(r"([A-Z])(\d+)$", c)
        if match:
            alphabet_part = match.group(1)
            numeric_part = match.group(2)
            return alphabet_part, int(numeric_part)
        
        else:
            if re.match(r"[A-Z]$", c):
                return c, 0
            
            else:
                if len(c.split(" ")) > 1:
                    temp_cabin_class = []
                    temp_cabin_number = []
                    for sp in c.split(" "):
                        sp.strip()
                        cabin_class, cabin_number = process_cabin(sp)
                        temp_cabin_class.append(cabin_class)
                        temp_cabin_number.append(cabin_number)
                    a_set = set(temp_cabin_class)
                    if len(a_set) == 1:
                        return a_set.pop(), int(np.median(temp_cabin_number))
                    else:
                        return "z", temp_cabin_number[-1]
    
    test_df['cabin_class'] = test_df['Cabin'].apply(process_cabin).apply(lambda x: x[0] if x is not None else None)
    
    # Fill missing cabin class using training data patterns
    def fill_missing_cabin_class(df, train_df_with_cabins):
        df_filled = df.copy()
        
        # Use training data to create fare-cabin class mapping
        fare_means = train_df_with_cabins.groupby(['Pclass', 'cabin_class'])['Fare'].mean()
        
        missing_mask = df['cabin_class'].isna()
        
        for idx in df[missing_mask].index:
            passenger_fare = df.loc[idx, 'Fare']
            passenger_pclass = df.loc[idx, 'Pclass']
            
            if pd.isna(passenger_fare):
                continue
            
            try:
                available_classes = fare_means[passenger_pclass].index
                
                fare_differences = {}
                for cabin_class in available_classes:
                    mean_fare = fare_means[passenger_pclass][cabin_class]
                    fare_differences[cabin_class] = abs(passenger_fare - mean_fare)
                
                if fare_differences:
                    best_match = min(fare_differences, key=fare_differences.get)
                    df_filled.loc[idx, 'cabin_class'] = best_match
                
            except KeyError:
                continue
        
        return df_filled
    
    # Create training cabin class data for reference
    train_with_cabin_class = train_df.copy()
    train_with_cabin_class['cabin_class'] = train_with_cabin_class['Cabin'].apply(process_cabin).apply(lambda x: x[0] if x is not None else None)
    train_cabin_data = train_with_cabin_class[train_with_cabin_class['cabin_class'].notna()]
    
    test_df = fill_missing_cabin_class(test_df, train_cabin_data)
    
    # 10. CABIN NUMBER
    print("10. Processing cabin numbers...")
    
    test_df["cabin_number"] = test_df["Cabin"].apply(process_cabin).apply(lambda x: x[1] if x is not None else None)
    
    # Fill missing cabin numbers using training data ranges
    def assign_missing_cabins_by_pclass_range(df, train_df_with_cabins):
        df_filled = df.copy()
        
        # Get cabin ranges from training data
        pclass_ranges = {}
        for pclass in train_df_with_cabins['Pclass'].unique():
            pclass_cabins = train_df_with_cabins[
                (train_df_with_cabins['Pclass'] == pclass) & 
                (train_df_with_cabins['cabin_number'].notna())
            ]['cabin_number']
            
            if len(pclass_cabins) > 0:
                pclass_ranges[pclass] = (int(pclass_cabins.min()), int(pclass_cabins.max()))
        
        # Fill missing values
        for pclass in pclass_ranges.keys():
            missing_mask = (df_filled['Pclass'] == pclass) & (df_filled['cabin_number'].isna())
            missing_indices = df_filled[missing_mask].index
            
            if len(missing_indices) > 0:
                min_cabin, max_cabin = pclass_ranges[pclass]
                random_cabins = [float(random.randint(min_cabin, max_cabin)) for _ in range(len(missing_indices))]
                df_filled.loc[missing_indices, 'cabin_number'] = random_cabins
        
        return df_filled
    
    # Create training cabin number data
    train_with_cabin_number = train_df.copy()
    train_with_cabin_number['cabin_number'] = train_with_cabin_number['Cabin'].apply(process_cabin).apply(lambda x: x[1] if x is not None else None)
    
    test_df = assign_missing_cabins_by_pclass_range(test_df, train_with_cabin_number)
    
    # 11. FINAL CLEANUP
    print("11. Final cleanup...")
    
    test_df = test_df.drop("Cabin", axis=1)
    
    # Convert cabin_class to numeric codes
    # Use training data to get consistent encoding
    train_cabin_classes = sorted(train_with_cabin_class['cabin_class'].dropna().unique())
    cabin_class_mapping = {cls: i for i, cls in enumerate(train_cabin_classes)}
    test_df['cabin_class'] = test_df['cabin_class'].map(cabin_class_mapping).fillna(-1)
    
    test_df = test_df.drop("ticket_no", axis=1)
    
    print("\nFinal test data shape:", test_df.shape)
    print("Missing values in processed test data:")
    print(test_df.isna().sum())
    
    return test_df

# Usage:
# processed_test_df = process_test_data("data/test.csv")  # Uses original train.csv by default
# OR specify custom training data path:
# processed_test_df = process_test_data("data/test.csv", "data/train.csv")