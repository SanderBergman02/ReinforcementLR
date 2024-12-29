import pm4py
from pm4py.objects.conversion.log import converter as log_conversion
from pm4py.objects.log.obj import EventLog, Trace
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter



# Data loading
file_path = "C://Users//Gebruiker//Downloads//BPI_Challenge_2017//BPI_Challenge_2017.csv"
data = pd.read_csv(file_path)
data['time'] = pd.to_datetime(data['time'])
bpi_log = pm4py.format_dataframe(data, case_id='case', activity_key='event', timestamp_key='time')
bpi_event_log = log_conversion.apply(bpi_log, {}, log_conversion.TO_EVENT_LOG)

# Remove "A_Pending"
bpi_pending = [len(list(filter(lambda e: e["concept:name"] == "A_Pending", trace))) > 0 for trace in bpi_event_log]
bpi_not_pending = [len(list(filter(lambda e: e["concept:name"] == "A_Pending", trace))) == 0 for trace in bpi_event_log]
print(f"Number of traces containing 'A_Pending': {sum(bpi_pending)}")
print(f"Number of traces not containing 'A_Pending': {sum(bpi_not_pending)}")

bpi_filtered_log = pm4py.filter_event_attribute_values(bpi_event_log, "concept:name", "A_Pending", level="event", retain=False)

# Create prefixes
bpi_prefixes = EventLog([Trace(trace[0:10], attributes=trace.attributes) for trace in bpi_filtered_log])
print(f"Number of traces in prefixes: {len(bpi_prefixes)}")



# Data preprocessing
bpi_df = pm4py.convert_to_dataframe(bpi_prefixes)
print(f"DataFrame shape: {bpi_df.shape}")
print(f"DataFrame columns: {bpi_df.columns}")
print(f"First few rows of DataFrame:\n{bpi_df.head()}")

# Bag of Words
bpi_case_act = bpi_df.loc[:, ["case:concept:name", "concept:name"]]
bpi_act_presence = bpi_case_act.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
bpi_bag = (bpi_act_presence > 0).astype(int).to_numpy()
print(f"Binary Bag of Words Shape: {bpi_bag.shape}")
print(f"Binary Bag of Words Sample:\n{bpi_bag[:5]}")

# Case attributes
case_attributes = bpi_df.groupby("case:concept:name").agg({"CreditScore": "first", "LoanGoal": "first"}).reset_index()

# Credit score feature
case_attributes["CreditScoreBinary"] = (case_attributes["CreditScore"] > 0).astype(int)
print(f"Case Attributes with Credit Score Binary:\n{case_attributes.head()}")

# Loan goal feature
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
loan_goal_one_hot = one_hot_encoder.fit_transform(case_attributes[["LoanGoal"]])
loan_goal_one_hot_columns = one_hot_encoder.get_feature_names_out(["LoanGoal"])
print(f"Loan Goal One-Hot Encoded Shape: {loan_goal_one_hot.shape}")
print(f"Loan Goal One-Hot Encoded Sample:\n{loan_goal_one_hot[:5]}")
print(f"Loan Goal One-Hot Encoded Columns: {loan_goal_one_hot_columns}")

# Combine everything
case_order = bpi_act_presence.index.tolist()
case_attributes = case_attributes.set_index("case:concept:name").reindex(case_order)
credit_score_binary = case_attributes["CreditScoreBinary"].to_numpy().reshape(-1, 1)
loan_goal_one_hot = loan_goal_one_hot[:len(case_order)]
bpi_bag_with_loangoal = np.hstack([bpi_bag, credit_score_binary, loan_goal_one_hot])

print(f"Feature set shape: {bpi_bag_with_loangoal.shape}")
print(f"Feature set sample:\n{bpi_bag_with_loangoal[:5]}")



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(bpi_bag_with_loangoal, bpi_pending, test_size=0.3, random_state=42)

from xgboost import XGBClassifier

model = XGBClassifier(
    # objective = 'binary:logistic',
    objective = 'multi:softprob', num_class=2,
    booster='gbtree', gamma=0,
    colsample_bylevel=1, colsample_bynode=1, colsamp_bytree=1,
    learning_rate=0.01, max_delta_step=0, max_depth=50,
    min_child_weight=1, validate_parameters=5,
    n_estimators=50, num_parallel_tree=5, random_state=0,
    reg_alpha=2, reg_lambda=1, subsample=1,
    importance_type='weight',
    tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
    nthread=-1
)

model.fit(X_train, y_train)

y_pred_rf = model.predict(X_test)

# rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_clf.fit(X_train, y_train)
#
# # Evaluate model
# y_pred_rf = rf_clf.predict(X_test)

preds = []
for i in y_pred_rf:
    if i[0] == 1:
        preds.append(False)
    else:
        preds.append(True)

print(y_test)
print(y_pred_rf[0])

# Train model
# rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_clf.fit(X_train, y_train)

# Evaluate model
# y_pred_rf = rf_clf.predict(X_test)

print("\nClassification report random forest:")
print(classification_report(y_test, preds))
print("\nConfusion matrix random forest:")
print(confusion_matrix(y_test, preds))
