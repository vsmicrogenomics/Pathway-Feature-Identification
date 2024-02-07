import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(susceptibility_path, kegg_path):
    susceptibility_df = pd.read_csv(susceptibility_path, sep='\t')
    kegg_df = pd.read_csv(kegg_path, sep='\t')
    return susceptibility_df, kegg_df

def preprocess_data(susceptibility_df, kegg_df):
    # Filter for completeness >= 80%
    kegg_df_filtered = kegg_df.loc[:, 'pathway group':].apply(pd.to_numeric, errors='coerce').ge(80).any(axis=1)
    kegg_df_filtered = kegg_df[kegg_df_filtered]

    # Reshape and merge
    kegg_long_df = pd.melt(kegg_df_filtered, id_vars=['module', 'name', 'pathway group'], var_name='Organism', value_name='Completeness')
    kegg_long_df = kegg_long_df[kegg_long_df['Completeness'] >= 80]
    merged_df = pd.merge(kegg_long_df, susceptibility_df, on='Organism', how='inner')

    return merged_df

def fit_logistic_regression(merged_df):
    merged_df['AMR_binary'] = np.where(merged_df['Category'] == 'Carbapenem-Resistant', 1, 0)
    module_presence = pd.pivot_table(merged_df, values='Completeness', index='Organism', columns='module', fill_value=0).ge(80).astype(int)
    amr_status = merged_df[['Organism', 'AMR_binary']].drop_duplicates().set_index('Organism')['AMR_binary']

    module_significance = []
    for module in module_presence.columns:
        X = module_presence[[module]]
        y = amr_status.loc[X.index]
        if X[module].std() == 0:
            continue
        model = LogisticRegression(solver='liblinear')
        model.fit(X, y)
        y_pred_prob = model.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y, y_pred_prob)
        module_significance.append({'Module': module, 'Coefficient': model.coef_[0][0], 'AUC': auc_score})
    
    return pd.DataFrame(module_significance)

def plot_results_with_corrected_legend(module_significance_df, merged_df, output_path):
    module_significance_df['abs_coefficient'] = module_significance_df['Coefficient'].abs()
    sorted_df = module_significance_df.sort_values(by='abs_coefficient', ascending=False)
    
    neg_df = sorted_df[sorted_df['Coefficient'] < 0].sort_values(by='Coefficient', ascending=True)
    pos_df = sorted_df[sorted_df['Coefficient'] > 0].sort_values(by='Coefficient', ascending=True)  # Change here for positive gradient
    
    sorted_df = pd.concat([neg_df, pos_df])
    
    neg_colors = sns.light_palette("red", n_colors=len(neg_df)+1, reverse=True).as_hex()[:-1]
    pos_colors = sns.light_palette("green", n_colors=len(pos_df)+1, reverse=False).as_hex()[1:]  # Reverse=False for the correct gradient
    gradient_colors = neg_colors + pos_colors
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Module', data=sorted_df, palette=gradient_colors)
    plt.title('Association of KEGG Modules with Carbapenem Resistance', fontweight='bold')
    plt.xlabel('Coefficient Value', fontweight='bold')
    plt.ylabel('KEGG Module', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.axvline(x=0, color='grey', linestyle='--')
    
    # Filter the legend DataFrame to include only the modules present in the plot
    filtered_legend_df = merged_df[merged_df['module'].isin(sorted_df['Module'].unique())]
    
    # Create a custom legend outside the plot
    # Extract unique module names and pathway groups for the legend
    legend_info = filtered_legend_df[['module', 'name', 'pathway group']].drop_duplicates().sort_values(by='module')
    legend_text = [f"{row['module']}: {row['name']} ({row['pathway group']})" for index, row in legend_info.iterrows()]
    
    # Place the legend to the right of the plot
    plt.legend(legend_text, title='Module Legend', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large', title_fontsize='large', frameon=False)
    
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the layout to make space for the legend
    plt.savefig(output_path, bbox_inches='tight')
    
    # Save module coefficients to a text file
    sorted_df[['Module', 'Coefficient']].to_csv(output_path.replace('.pdf', '_coefficients.txt'), sep='\t', index=False)

def evaluate_model(merged_df, module_significance_df):
    # Identify the column containing organism names
    organism_column = merged_df.columns[merged_df.columns.str.lower().str.contains('organism')][0]
    
    # Prepare data
    module_presence = pd.pivot_table(merged_df, values='Completeness', index=organism_column, columns='module', fill_value=0).ge(80).astype(int)
    amr_status = merged_df[[organism_column, 'AMR_binary']].drop_duplicates().set_index(organism_column)['AMR_binary']

    # Prepare features and target
    X = module_presence[module_significance_df['Module']]
    y = amr_status.loc[X.index]

    # Fit logistic regression model
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)

    # Predict probabilities
    y_pred_prob = model.predict_proba(X)[:, 1]

    # Predict classes
    y_pred = model.predict(X)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_prob)
    confusion_mat = confusion_matrix(y, y_pred)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(output_path.replace('.pdf', '_roc_curve.pdf'), bbox_inches='tight')  # Save ROC curve as PDF
    plt.show()

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC Score': roc_auc,
        'Confusion Matrix': confusion_mat
    }

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    input_folder = 'input'  # Update this to your input folder path
    output_folder = 'output'  # Update this to your output folder path
    
    susceptibility_path = f'{input_folder}/susceptibility_groups.txt'
    kegg_path = f'{input_folder}/OUT-BOTH-Cluster_module_completeness.tab'
    
    susceptibility_df, kegg_df = load_data(susceptibility_path, kegg_path)
    merged_df = preprocess_data(susceptibility_df, kegg_df)
    module_significance_df = fit_logistic_regression(merged_df)
    output_path_corrected_legend = f'{output_folder}/module_association_plot_corrected_legend.pdf'
    plot_results_with_corrected_legend(module_significance_df, merged_df, output_path_corrected_legend)
    
    # Evaluate the model
    evaluation_results = evaluate_model(merged_df, module_significance_df)

    # Print evaluation results
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value}")
        else:
            print("\nConfusion Matrix:")
            print(value)
            
    # Write evaluation results to a text file
    with open(f'{output_folder}/evaluation_results.txt', 'w') as f:
        f.write("Evaluation Results:\n")
        for metric, value in evaluation_results.items():
            if metric != 'Confusion Matrix':
                f.write(f"{metric}: {value}\n")
            else:
                f.write("\nConfusion Matrix:\n")
                f.write(f"{value}\n")
