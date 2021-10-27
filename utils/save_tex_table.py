import numpy as np


def save_tex_table(rows_list, filepath, use_hline=True):
    with open(filepath, 'w') as f:
        put_table_to_file(rows_list, f, use_hline)


def put_table_to_file(rows_list, file, use_hline):
    table_header = "\\begin{tabular}{|r|" + "l|"*(len(rows_list[0])-1) + "}\n"
    file.write(table_header)
    file.write("  \\hline\n")

    for row in rows_list:
        save_single_row(row, file, use_hline)

    file.write("\end{tabular}\n")


def save_single_row(row, file, use_hline):
    row_string = get_row_string(row)
    string_to_save = row_string[:-2] + "\\\\ \n"
    file.write(string_to_save)
    if use_hline:
        file.write("  \\hline\n")


def get_row_string(row):
    row_string = "  "
    for elem in row:
        row_string = row_string + sanitize(elem) + " & "

    return row_string


def sanitize(elem):
    if type(elem) in (np.float16, np.float32, np.float64, np.float_, float):
        elem = sanitize_float(elem)
    return str(elem).replace("_", " ")


def sanitize_float(number, decimal_places=3):
    return crop_to_decimal_places(str(number), decimal_places=decimal_places)


def crop_to_decimal_places(number_str, decimal_places=3):
    dot_position = number_str.find('.')
    if number_str.count('e-') == 0:  # normal dot notation. not 1.0e-4
        return number_str[:dot_position+decimal_places+1]
    else:
        e_position = number_str.find('e')
        return number_str[:dot_position+4] + number_str[e_position:]
