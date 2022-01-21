import numpy as np
import scipy.stats
import pathlib
import prettytable
import utils.save_tex_table
import os


def get_index_of_given_model(model_name, model_list):
    for i in range(len(model_list)):
        if(model_list[i] == model_name):
            return i
    #print(f'indexing problem! {model_name} not in {model_list}!')
    return -1

def convert_table_to_dict(table):
    result_dict = {}
    for i in range(len(table)-1):
        result_dict[table[i+1][0]] = table[i+1][1]
    return result_dict

def is_significant(value):
    if(value != None and value < 0.05):
        return True
    return False

def get_significance_table_from_pvalue(pvalue, list_of_models, attribute_name):
    significance_table = {}
    for model_name in list_of_models:
        significance_table[model_name] = {}
        extractor_index = get_index_of_given_model(model_name+'_' + attribute_name, pvalue[0])
        for comparison_model in list_of_models:
            comparison_index = get_index_of_given_model(comparison_model+'_' + attribute_name, pvalue[0])
            significance_table[model_name][comparison_model] = is_significant(pvalue[extractor_index][comparison_index])
    #print("SIGNIFICANCE: ", significance_table)
    return significance_table

def get_significance_table_from_pvalue_for_simple_model(pvalue, list_of_models, attribute_name, dataset_name):
    significance_table = {}
    for model_name in list_of_models:
        significance_table[model_name] = {}
        extractor_index = get_index_of_given_model(f'{model_name}_{dataset_name}_{attribute_name}', pvalue[0])
        #print(f'Extractor index {extractor_index}')
        for comparison_model in list_of_models:
            comparison_index = get_index_of_given_model(f'{comparison_model}_{dataset_name}_{attribute_name}', pvalue[0])
            #print(f'Compared index {comparison_index}')
            #print(f'Comparison of {model_name} with {comparison_model}')
            if (extractor_index != -1) or (comparison_index != -1):
                significance_table[model_name][comparison_model] = is_significant(pvalue[extractor_index][comparison_index])
            else:
                significance_table[model_name][comparison_model] = False
    #print("SIGNIFICANCE: ", significance_table)
    return significance_table

def strip_model_name(name):
    result = ""
    for word in name.split(" "):
        result+=word[:4] + " "
    splitted_result = result.split(" ")
    if(len(splitted_result) > 3):
        new_result = splitted_result[0][0] + splitted_result[1][0] + splitted_result[2][0]
        for rest in splitted_result[3:]:
            new_result += " " + rest
        result = new_result
    return result

def print_latex_results(list_of_datasets, attribute_name, list_of_models, accuracies, statistical_restult):
    print("\\begin{table}[]")
    print("\centering")
    print("\caption{Results for ", attribute_name, " attribute}")
    print("\\begin{tabular}{|l|", end="")
    for i in range(len(list_of_models)):
        print("c|", end="")
    print("}")

    print("\hline")
    print("Extraction method ", end="")
    for model in list_of_models:
        print(" & ", strip_model_name(model.replace("_", " ")), end="")
    print("  \\\\")
    print("\hline")
    for dataset in list_of_datasets:
        print(dataset.replace("_", " "), end="")
        for i, model in enumerate(list_of_models):
            print(" & ", round(accuracies[dataset][i], 3), end="")
        print("  \\\\")
        for i, model in enumerate(list_of_models):
            print(" & \small{", end="")
            worse_models = statistical_restult[dataset][i]
            if len(worse_models) == 0:
                print("-", end="")
            else:
                for worse_model in worse_models:
                    print(worse_model, end=" ")
            print("} ", end="")
        print("  \\\\")
        print("\hline")
    print("\end{tabular}")
    print("\end{table}")

def map_title_to_table(numpy_filename):
    translation_dict = {"title":"table1",
                        "text":"table2",
                        "title_4M":"table3",
                        "text_4M":"table4",
                        "title_12M":"table5",
                        "text_12M":"table6",
                        "text_ensemble_avrg_3M":"table7",
                        "text_concat_extraction_model_avrg_mutual_info_3M":"table8",
                        "text_concat_extraction_model_avrg_anova_3M":"table9",
                        "text_concat_extraction_model_avrg_pca_3M":"table10",
                        "title_ensemble_avrg_3M":"table11",
                        "title_concat_extraction_model_avrg_mutual_info_3M":"table12",
                        "title_concat_extraction_model_avrg_anova_3M":"table13",
                        "title_concat_extraction_model_avrg_pca_3M":"table14",
    }
    return translation_dict[numpy_filename]

def save_tables_to_numpy(list_of_datasets, list_of_models, accuracies, title):
    print(f"creating tensor of shape ({len(list_of_datasets)},{len(list_of_models)}) with results:")
    result_tensor = np.zeros((len(list_of_datasets), len(list_of_models)))
    for i, dataset in enumerate(list_of_datasets):
        for j, model in enumerate(list_of_models):
            result_tensor[i][j] = accuracies[dataset][j]
    print(result_tensor)
    title = map_title_to_table(title)
    pred_filename = f'./tables_numpy/{title}.npy'
    os.makedirs(os.path.dirname(pred_filename), exist_ok=True)
    np.save(pred_filename, result_tensor)

def print_latex_3M_average_table(list_of_datasets, attribute_name, list_of_models, accuracies):
    print("\\begin{table}[]")
    print("\centering")
    print("\caption{Results for ", attribute_name, " attribute}")
    print("\\begin{tabular}{|l|", end="")
    for i in range(len(list_of_models)):
        print("c|", end="")
    print("}")

    print("\hline")
    print("Extraction method ", end="")
    for model in list_of_models:
        print(" & ", strip_model_name(model.replace("_", " ")), end="")
    print("  \\\\")
    print("\hline")
    for dataset in list_of_datasets:
        print(dataset.replace("_", " "), end="")
        for i, model in enumerate(list_of_models):
            print(" & ", round(accuracies[dataset][i], 3), end="")
        print("  \\\\")
        print("\hline")
    print("\end{tabular}")
    print("\end{table}")

def perform_statistical_analysis(results, accuracies, attribute_name, list_of_models, mode=""):
    statistical_restult = {}
    list_of_datasets = ('bs_detector', 'esp_fake', 'mixed')
    for dataset_name in list_of_datasets:
        _, pvalue = statistical_tests_table(results, dataset_name)
        #pretty_print_table(pvalue)
        statistical_restult[dataset_name] = []
        accuracies_for_dataset = accuracies[dataset_name]
        if(mode != ""):
            significance_table = get_significance_table_from_pvalue(pvalue, list_of_models, attribute_name)
        else:
            significance_table = get_significance_table_from_pvalue_for_simple_model(pvalue, list_of_models, attribute_name, dataset_name)
        for i, extractor_name in enumerate(list_of_models):
            current_accuracy = accuracies_for_dataset[i]
            statistical_restult[dataset_name].append([])
            for j, extractor_to_be_compared in enumerate(list_of_models):
                compared_extractor_accuracy = accuracies_for_dataset[j]
                if(significance_table[extractor_name][extractor_to_be_compared]):
                    #print("There is statistical difference between", extractor_name, "and ", extractor_to_be_compared)
                    if(compared_extractor_accuracy != 0 and current_accuracy > compared_extractor_accuracy):
                        #print("Because current extractor (",extractor_name,") has better accuracy (", current_accuracy, " vs ", compared_accuracy, ") - we mark second one (",extractor_to_be_compared,") as worse - index ", j+1)
                        statistical_restult[dataset_name][i].append(j+1)
    print("RESULTS FOR ", attribute_name, "ATTRIBUTE:")
    print("Models", list_of_models)
    print("Accuracies", accuracies)
    print("Worse models list", statistical_restult)
    print_latex_results(list_of_datasets, attribute_name, list_of_models, accuracies, statistical_restult)
    save_tables_to_numpy(list_of_datasets, list_of_models, accuracies, f"{attribute_name}")
    return statistical_restult

def get_dataset_row_of_accuracies_for_all_models(avrg_table, dataset, list_of_models, attribute):
    row = []
    for model_name in list_of_models:
        if f'{model_name}_{attribute}' in avrg_table[dataset].keys():
                row.append(avrg_table[dataset][f'{model_name}_{attribute}'])
        elif f'{model_name}_{dataset}_{attribute}' in avrg_table[dataset].keys():
                row.append(avrg_table[dataset][f'{model_name}_{dataset}_{attribute}'])
        else:
            row.append(0)
    return row

def add_mode_to_attribute(attribute, mode):
    if mode != "":
        return f'{attribute}_{mode}'
    return attribute

def filter_results(results, list_of_datasets, list_of_models, modes):
    key_list = []
    for key in results.keys():
        for dataset in list_of_datasets:
            for model in list_of_models:
                for mode in modes:
                    if(mode != ""):
                        if (f'{dataset}_{model}' in key) and (mode in key):
                                key_list.append(key)
                    elif (f'{dataset}_{model}_{dataset}' in key) and (key.split("_")[-1] == "title" or key.split("_")[-1] == "text"):
                            key_list.append(key)
    result_dictionary = { key: results[key] for key in key_list}
    #print(result_dictionary)
    return result_dictionary

def show_average_table_for_3M(results, list_of_datasets, list_of_models, attribute):
    result_dict = {}
    print("------")
    print(f'RESULTS for {attribute} attribute:')
    models_result_average_on_dataset = ['tf_idf', 'lda', 'bert_multi', 'bert_eng', 'beto']
    print("columns: ", models_result_average_on_dataset)
    for model_name in list_of_models:
        for dataset_name in list_of_datasets:
            models = {
            'esp_fake': ('bert_multi', 'beto', 'lda', 'tf_idf'),
            'bs_detector': ('bert_eng', 'bert_multi', 'lda', 'tf_idf'),
            'mixed': ('bert_eng', 'bert_multi', 'beto', 'lda', 'tf_idf'),
            }
            models_order = models[dataset_name]
            dataset_dict={}
            for i, model in enumerate(models_order):
                dataset_dict[model] = np.mean(results[f'{dataset_name}_{model_name}_{attribute}_3M'][i*10:((i+1)*10)])
            result_dict[dataset_name]=dataset_dict
        print(f'{model_name} results:')
        for  dataset in list_of_datasets:
            print(f'{dataset}:  ', end =" ")
            for model in models_result_average_on_dataset:
                if model not in result_dict[dataset].keys():
                    result_dict[dataset][model] = 0
                    print(0, end=" ")
                else:
                    print(result_dict[dataset][model], end=" ")
            print(" ")
        accuracies_dict = {}
        for dataset in list_of_datasets:
            accuracies_dict[dataset] = [ result_dict[dataset][model] for model in models_result_average_on_dataset]
        print(accuracies_dict)
        print_latex_3M_average_table(list_of_datasets, (attribute+f" {model_name}").replace("_"," "), models_result_average_on_dataset, accuracies_dict)
        save_tables_to_numpy(list_of_datasets, models_result_average_on_dataset, accuracies_dict, f"{attribute}_{model_name}_3M")

def perform_statistical_analysis_based_on_results(results, list_of_datasets, list_of_models, mode=""):
    avrg_table = {}
    for dataset in list_of_datasets:
        avr_table_dataset = get_average_table(results, dataset)
        avrg_table[dataset] = convert_table_to_dict(avr_table_dataset)
    
    title_results = {}
    text_results = {}
    title_attribute = add_mode_to_attribute("title", mode)
    text_attribute = add_mode_to_attribute("text", mode)
    for dataset in list_of_datasets:
        title_results[dataset] = get_dataset_row_of_accuracies_for_all_models(avrg_table, dataset, list_of_models, title_attribute)
        text_results[dataset] = get_dataset_row_of_accuracies_for_all_models(avrg_table, dataset, list_of_models, text_attribute)
    
    perform_statistical_analysis(results, text_results, text_attribute, list_of_models, mode)
    perform_statistical_analysis(results, title_results, title_attribute, list_of_models, mode)

def main():
    results = load_results()
    os.makedirs('tables/', exist_ok=True)
    
    original_results = results
    list_of_datasets = ['bs_detector', 'esp_fake', 'mixed']

    print("MODEL COMPARISON")
    mode = ""
    list_of_models = ['tf_idf', 'lda', 'bert_multi', 'bert_eng', 'beto']
    results = filter_results(original_results, list_of_datasets, list_of_models, [mode])
    perform_statistical_analysis_based_on_results(results, list_of_datasets, list_of_models, mode)

    print("==============================================")
    print("4M ENSEMBLE")
    mode = "4M"
    list_of_models = ['ensemble_avrg', 'concat_extraction_model_avrg_mutual_info', 'concat_extraction_model_avrg_anova', 'concat_extraction_model_avrg_pca']
    results = filter_results(original_results, list_of_datasets, list_of_models, [mode])
    perform_statistical_analysis_based_on_results(results, list_of_datasets, list_of_models, mode)

    print("==============================================")
    print("12M ENSEMBLE")
    mode = "12M"
    list_of_models = ['ensemble_avrg', 'concat_extraction_model_avrg_mutual_info', 'concat_extraction_model_avrg_anova', 'concat_extraction_model_avrg_pca']
    results = filter_results(original_results, list_of_datasets, list_of_models, [mode])
    perform_statistical_analysis_based_on_results(results, list_of_datasets, list_of_models, mode)
    
    print("==============================================")
    print("3M ENSEMBLE")
    mode = "3M"
    list_of_models = ['ensemble_avrg', 'concat_extraction_model_avrg_mutual_info', 'concat_extraction_model_avrg_anova', 'concat_extraction_model_avrg_pca']
    results = filter_results(original_results, list_of_datasets, list_of_models, [mode])
    show_average_table_for_3M(results, list_of_datasets, list_of_models, "text")
    show_average_table_for_3M(results, list_of_datasets, list_of_models, "title")

def load_results():
    results = {}
    results_path = pathlib.Path('results/')
    for filepath in results_path.iterdir():
        name = filepath.name.split('.')[0]
        res = np.load(filepath)
        results[name] = res
    return results


def get_average_table(results, dataset_name):
    table = [['Extraction method', 'Average accuracy']]
    for name, res in results.items():
        if name.startswith(dataset_name):
            extraction_name = name[len(dataset_name)+1:]
            avrg_acc = sum(res) / len(res)
            table.append([extraction_name, avrg_acc])
    return table


def pretty_print_table(table, table_name=None):
    pretty_table = prettytable.PrettyTable(table[0])
    pretty_table.add_rows(table[1:])
    if table_name:
        pretty_table.title = table_name
    print(pretty_table)


def statistical_tests_table(results, dataset_name):
    selected_results = [(key[len(dataset_name)+1:], value) for key, value in results.items() if key.startswith(dataset_name)]
    if (selected_results[0][1].shape == (40,)) or (selected_results[0][1].shape == (50,)):
        print("3M detected! - skipping analysis!")
        exit(0)

    else:
        table_stat = [[None]]
        table_pvalue = [[None]]
        for i in range(len(selected_results)):
            row_stat = [selected_results[i][0]]
            row_pvalue = [selected_results[i][0]]
            for j in range(len(selected_results)):
                if len(table_stat[0]) < len(selected_results)+1:
                    table_stat[0].append(f'{selected_results[j][0]}')
                    table_pvalue[0].append(f'{selected_results[j][0]}')
                if i == j:
                    row_stat.append(None)
                    row_pvalue.append(None)
                    continue
                f, p = cv52cft(selected_results[i][1], selected_results[j][1])
                row_stat.append(f)
                row_pvalue.append(p)
            table_stat.append(row_stat)
            table_pvalue.append(row_pvalue)
        return table_stat, table_pvalue


def cv52cft(a, b):
    """5x2cv combined F test
    Args:
        - a (ndarray) numpy array with results from 5x2 cv folds
        - b (ndarray) numpy array with results from 5x2 cv folds
    """
    d = a.reshape(2, 5) - b.reshape(2, 5)
    f = np.sum(np.power(d, 2)) / (2 * np.sum(np.var(d, axis=0, ddof=0)))
    p = 1-scipy.stats.f.cdf(f, 10, 5)
    return f, p


if __name__ == '__main__':
    main()
