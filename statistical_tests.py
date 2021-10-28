import numpy as np
import scipy.stats
import pathlib
import prettytable
import utils.save_tex_table
import os


def main():
    results = load_results()

    os.makedirs('tables/', exist_ok=True)

    avrg_table_esp = get_average_table(results, 'esp_fake')
    pretty_print_table(avrg_table_esp, table_name='Esp fake avrg results')
    utils.save_tex_table.save_tex_table(avrg_table_esp, 'tables/esp_fake_avrg.tex')
    avrg_table_bs_detector = get_average_table(results, 'bs_detector')
    pretty_print_table(avrg_table_bs_detector, table_name='bs_detector avrg results')
    utils.save_tex_table.save_tex_table(avrg_table_bs_detector, 'tables/bs_detector_avrg.tex')

    stats_table_esp = statistical_tests_table(results, 'esp_fake')
    pretty_print_table(stats_table_esp, table_name='Esp fake F-test')
    utils.save_tex_table.save_tex_table(stats_table_esp, 'tables/esp_stats.tex')
    stats_table_bs_detector = statistical_tests_table(results, 'bs_detector')
    pretty_print_table(stats_table_bs_detector, table_name='bs_detector F-test')
    utils.save_tex_table.save_tex_table(stats_table_bs_detector, 'tables/bs_detector_stats.tex')


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
    selected_results = [(key, value) for key, value in results.items() if key.startswith(dataset_name)]

    table = [[None]]
    for i in range(len(selected_results)):
        row = [selected_results[i][0]]
        for j in range(len(selected_results)):
            if len(table[0]) < 2 * len(selected_results):
                table[0].append(f'{selected_results[j][0]}_stat')
                table[0].append(f'{selected_results[j][0]}_p-value')
            if i == j:
                row.append(None)
                row.append(None)
                continue
            f, p = cv52cft(selected_results[i][1], selected_results[j][1])
            row.append(f)
            row.append(p)
        table.append(row)
    return table


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
