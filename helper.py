import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from embanded.embanded.embanded_numpy import EMBanded
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
#from group_lasso import GroupLasso
import ast
from sklearn.metrics import accuracy_score


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def evaluate_auc(y_true,y_pred,labels):
    list_auc = []
    mean_auc = 0
    for label in labels:
        auc_score = roc_auc_score(y_true[label],y_pred[label])
        list_auc.append(auc_score)
        mean_auc += auc_score
    return mean_auc/len(labels)

def get_xgboost_imp_multi(multilabel_model, feature_names):
    aggregated_importance = {feature: [0] for feature in feature_names}
    for i, estimator in enumerate(multilabel_model.estimators_):
        # Get booster from the current XGBClassifier
        booster = estimator.get_booster()
        booster.feature_names = feature_names

        # Retrieve importance based on gain for this model
        importance_dict = booster.get_score(importance_type='gain')
        
        # Update aggregated importance
        for feature, importance in importance_dict.items():
            if feature in aggregated_importance:
                aggregated_importance[feature].append(importance)
            else:
                aggregated_importance[feature] = [importance]
    # Compute average importance for each feature
    average_importance = {feature: sum(values) / len(values) for feature, values in aggregated_importance.items()}
    # Convert to DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': list(average_importance.keys()),
        'Average Gain': list(average_importance.values())
    }).sort_values(by='Average Gain', ascending=False)

    importance_df.set_index('Feature', inplace=True)

    return importance_df

def get_xgboost_imp_single(model, feature_names):
    """
    Compute feature importance for a trained XGBClassifier.
    
    Args:
        model (XGBClassifier): Trained XGBoost model.
        feature_names (list): List of feature names.

    Returns:
        pd.DataFrame: Feature importance sorted by average gain.
    """
    booster = model.get_booster()
    booster.feature_names = feature_names  # Assign feature names

    # Retrieve feature importance based on gain
    importance_dict = booster.get_score(importance_type='gain')

    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Average Gain': list(importance_dict.values())
    })

    # Set 'Feature' as the index
    importance_df.set_index('Feature', inplace=True)

    # Reindex the DataFrame using the complete feature_names list,
    # filling any missing values with 0
    importance_df = importance_df.reindex(feature_names, fill_value=0)

    # Sort the DataFrame by 'Average Gain' in descending order
    importance_df = importance_df.sort_values(by='Average Gain', ascending=False)

    return importance_df

def evaluate_auc_xgboost(clf,x_te,x_tr,y_tr,y_te,labels,weird_probas=None):
  preds_te = clf.predict_proba(x_te)
  preds_tr = clf.predict_proba(x_tr)
  list_auc_train= list()
  list_auc_test = list()
  mean_auc_train = 0
  mean_auc_test = 0
  for i in range(len(labels)):
    if weird_probas==True:
      p_tr = preds_tr[:,i]
      p_te = preds_te[:,i]
    else:
      p_tr = preds_tr[i][:,1]
      p_te = preds_te[i][:,1]
    auc_train = roc_auc_score(y_tr[labels[i]],p_tr)
    list_auc_train.append(auc_train)
    auc_test = roc_auc_score(y_te[labels[i]],p_te)
    list_auc_test.append(auc_test)
    mean_auc_train += auc_train
    mean_auc_test += auc_test

  mean_auc_train = mean_auc_train/len(labels)
  mean_auc_test = mean_auc_test/len(labels)

  return mean_auc_train, mean_auc_test


def get_group_imp(df, groups):
    group_imp = {}
    for k,v in groups.items():
        group_imp[k] = df.loc[v]['Average Gain'].sum()/len(df.loc[v])
    
    norm = sum(group_imp.values())
    for k,v in group_imp.items():
        group_imp[k] = group_imp[k]/norm

    return group_imp

def train_eval_embanded_multi(X_train, y_train, X_test, y_test, partitions, partition_name, labels_dict, model_name, results):
    # Train
    feats_train = [X_train[v].to_numpy() for k,v in partitions[partition_name].items()]
    all_models = {}
    num_of_targets = len(labels_dict.keys())
    num_of_groups = len(partitions[partition_name])
    lambdas = np.zeros(num_of_groups)
    for k,v in labels_dict.items():
        model = EMBanded()
        y_n = y_train[v].to_numpy().astype('float64').reshape(-1,1)
        model.fit(feats_train,y_n) 
        all_models[v] = model
        lambdas += np.array(model.lambdas)
    lambdas = lambdas / num_of_targets

    # Predict
    result = {'model':model_name, 'partition':partition_name}
    d={}
    for i,group in enumerate(list(partitions[partition_name].keys())):
        d[group] = lambdas[i]/sum(lambdas)
    result['importance'] = str(d)
    for set_type, [X_set, y_set] in {'train':[X_train,y_train],'test':[X_test,y_test]}.items():
        feats = [X_set[v].to_numpy() for k,v in partitions[partition_name].items()]
        df = pd.DataFrame()
        for label in all_models:
            df[label] = all_models[label].predict(feats).flatten()
        df = df.apply(softmax, axis=1)
        # Score
        score  = evaluate_auc(y_set,df,df.columns)
        result[set_type] = [score]
    results = pd.concat([results, pd.DataFrame(data=result)])

    return results

def train_eval_embanded_single(X_train, y_train, X_test, y_test, partitions, partition_name, labels_dict, model_name, results):
    # Train
    feats_train = [X_train[v].to_numpy() for k,v in partitions[partition_name].items()]
    all_models = {}
    num_of_targets = len(labels_dict.keys())
    num_of_groups = len(partitions[partition_name])
    lambdas = np.zeros(num_of_groups)

    for label in np.unique(y_train):
        model = EMBanded()
        ind = np.where(y_train==label)[0]
        y_n = np.zeros((y_train.shape[0],1))
        y_n[ind] = 1
        model.fit(feats_train,y_n) 
        all_models[label] = model
        lambdas += np.array(model.lambdas)
    lambdas = lambdas / num_of_targets

    # Predict
    result = {'model':model_name, 'partition':partition_name}
    d={}
    for i,group in enumerate(list(partitions[partition_name].keys())):
        d[group] = lambdas[i]/sum(lambdas)
    result['importance'] = str(d)
    for set_type, [X_set, y_set] in {'train':[X_train,y_train],'test':[X_test,y_test]}.items():
        feats = [X_set[v].to_numpy() for k,v in partitions[partition_name].items()]
        df = pd.DataFrame()
        for label in all_models:
            df[label] = all_models[label].predict(feats).flatten()
        df['y_pred'] = df.idxmax(axis=1)
        y_pred = df['y_pred'].to_numpy()
        # Score
        score  = accuracy_score(y_set,y_pred)
        result[set_type] = [score]
    results = pd.concat([results, pd.DataFrame(data=result)])

    return results


def train_eval_xgboost_multi(X_train, y_train, X_test, y_test, partition, partition_name, model_name, results):
    feats = [item for sublist in list(partition.values()) for item in sublist]
    xgb_estimator = xgb.XGBClassifier(max_depth=3,learning_rate = 0.1,objective='binary:logistic',eval_metric='auc', n_estimators = 50, gamma=7.5628225927223830,min_child_weight=6)
    multilabel_model = MultiOutputClassifier(xgb_estimator)
    multilabel_model.fit(X_train[feats], y_train,verbose=True)

    auc_train, auc_test = evaluate_auc_xgboost(multilabel_model,X_test[feats],X_train[feats],y_train,y_test,labels)
    imp_df = get_xgboost_imp_multi(multilabel_model, feats)
    group_imp = get_group_imp(imp_df, partition)
    results = pd.concat([results,pd.DataFrame({'model':[model_name],'partition':[partition_name],'importance':[str(group_imp)],'test':[auc_test],'train':[auc_train]})],ignore_index=True)

    return results

def train_eval_xgboost_single(X_train, y_train, X_test, y_test, partition, partition_name, model_name, results):
    feats = [item for sublist in list(partition.values()) for item in sublist]

    xgb_estimator = xgb.XGBClassifier(max_depth=3,eta= 0.3,objective='multi:softmax',num_class=9,importance_type='weight')
    xgb_estimator.fit(X_train[feats], y_train,verbose=True)
    acc_train = xgb_estimator.score(X_train[feats],y_train)
    acc_test = xgb_estimator.score(X_test[feats],y_test)

    imp_df = get_xgboost_imp_single(xgb_estimator, feats)
    group_imp = get_group_imp(imp_df, partition)
    results = pd.concat([results,pd.DataFrame({'model':[model_name],'partition':[partition_name],'importance':[str(group_imp)],'test':[acc_test],'train':[acc_train]})],ignore_index=True)
    
    return results

def format_results(results):
    # Flatten the importance dict into separate columns
    results['importance'] = results['importance'].apply(ast.literal_eval)
    importance_df = results['importance'].apply(pd.Series)
    df = pd.concat([results.drop(columns=['importance']), importance_df], axis=1)

    return df, importance_df

def plot_results(df, importance_df, partitions, dataset, approach=None, save=True):
    # Extract unique partitions, models, and features
    partitions_names = df['partition'].unique()
    models = df['model'].unique()
    features = importance_df.columns

    # Define the number of rows needed based on the number of partitions and columns per row
    n_cols = 2  # Number of columns per row
    n_rows = (len(partitions_names) + n_cols - 1) // n_cols  # Calculate the required rows

    # Set up the plots with multiple rows and 2 columns
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows), sharey=True)

    # Flatten the axs array for easy indexing if needed
    axs = axs.ravel()

    # Loop through each partition to create a plot
    for i, partition in enumerate(partitions_names):
        ax = axs[i]
        width = 0.2  # Width of each bar

        # Filter data for the current partition
        partition_data = df[df['partition'] == partition]

        features = list(partitions[partition].keys())

        # Generate positions for each feature group
        x_positions = np.arange(len(features))

        for j, model in enumerate(models):
            # Filter data for the current model
            model_data = partition_data[partition_data['model'] == model]
            y_values = model_data[features].values.flatten()

            # Bar positions
            bar_positions = x_positions + (j - len(models)/2) * width

            # Plot bars for the current partition and model
            ax.bar(bar_positions, y_values, width=width, label=f"{model}-{np.round(model_data['test'].values[0]*100, 2)}")

        ax.set_title(f"Partition: {partition}")
        ax.set_ylabel("Importance")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(features, rotation=45)
        ax.legend(title="Model")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide unused subplots if any
    for i in range(len(partitions_names), len(axs)):
        axs[i].axis('off')

    # Common layout adjustments
    plt.tight_layout()

    if save:
        plt.savefig(f'{dataset}_importance_{approach}.png')

    plt.show()