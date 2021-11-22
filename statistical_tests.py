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

def perform_statistical_analysis(results, accuracies, attribute_name, list_of_models):
    statistical_restult = {}
    for dataset_name in ('bs_detector', 'esp_fake', 'mixed'):
        _, pvalue = statistical_tests_table(results, dataset_name)
        statistical_restult[dataset_name] = []
        accuracies_for_dataset = accuracies[dataset_name]
        significance_table = get_significance_table_from_pvalue(pvalue, list_of_models, attribute_name)
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
    return statistical_restult

def get_dataset_row_of_accuracies_for_all_models(avrg_table, dataset, list_of_models, attribute):
    row = []
    for model_name in list_of_models:
        if f'{model_name}_title' in avrg_table[dataset].keys():
            row.append(avrg_table[dataset][f'{model_name}_{attribute}'])
        else:
            row.append(0)
    return row

def perform_statistical_analysis_based_on_results(results, list_of_datasets, list_of_models):
    avrg_table = {}
    for dataset in list_of_datasets:
        avr_table_dataset = get_average_table(results, dataset)
        avrg_table[dataset] = convert_table_to_dict(avr_table_dataset)
    
    title_results = {}
    text_results = {}
    for dataset in list_of_datasets:
        title_results[dataset] = get_dataset_row_of_accuracies_for_all_models(avrg_table, dataset, list_of_models, 'title')
        text_results[dataset] = get_dataset_row_of_accuracies_for_all_models(avrg_table, dataset, list_of_models, 'text')
    
    perform_statistical_analysis(results, text_results, 'text', list_of_models)
    perform_statistical_analysis(results, title_results, 'title', list_of_models)

def main():
    results = load_results()

    os.makedirs('tables/', exist_ok=True)

    avrg_table_esp = get_average_table(results, 'esp_fake')
    pretty_print_table(avrg_table_esp, table_name='Esp fake avrg results')
    utils.save_tex_table.save_tex_table(avrg_table_esp, 'tables/esp_fake_avrg.tex')
    avrg_table_bs_detector = get_average_table(results, 'bs_detector')
    pretty_print_table(avrg_table_bs_detector, table_name='bs_detector avrg results')
    utils.save_tex_table.save_tex_table(avrg_table_bs_detector, 'tables/bs_detector_avrg.tex')
    avrg_table_mixed = get_average_table(results, 'mixed')
    pretty_print_table(avrg_table_mixed, table_name='mixed avrg results')
    utils.save_tex_table.save_tex_table(avrg_table_mixed, 'tables/mixed_avrg.tex')

    for dataset_name in ('esp_fake', 'bs_detector', 'mixed'):
        stats, pvalue = statistical_tests_table(results, dataset_name)
        pretty_print_table(stats, table_name=f'{dataset_name} F-test stat')
        pretty_print_table(pvalue, table_name=f'{dataset_name} F-test pvalue')
        utils.save_tex_table.save_tex_table(stats, f'tables/{dataset_name}_stats.tex')
        utils.save_tex_table.save_tex_table(pvalue, f'tables/{dataset_name}_pvalue.tex')
    
    list_of_datasets = ['bs_detector', 'esp_fake', 'mixed']

    list_of_models = ['tf_idf', 'lda', 'bert_multi', 'bert_eng', 'beto']
    perform_statistical_analysis_based_on_results(results, list_of_datasets, list_of_models)
    
    list_of_models = ['ensemble_avrg', 'concat_extraction_model_avrg']
    perform_statistical_analysis_based_on_results(results, list_of_datasets, list_of_models)

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
