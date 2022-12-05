import matplotlib.pyplot as plt
import seaborn as sns


def main():
    extractors = ['tf-idf', 'lda', 'bert-multi', 'bert-eng', 'beto']
    base_text = {
        'kaggle': [.866, .911, .981, .985, None],
        'esp fake': [.751, .618, .786, None, .853],
        'mixed': [.729, .768, .972, .974, .982],
    }
    base_title = {
        'kaggle': [.904, .829, .958, .963, None],
        'esp fake': [.540, .553, .675, None, .852],
        'mixed': [.646, .631, .942, .948, .982],
    }

    bar_plot(extractors, base_text, 'text')
    plt.savefig('base_text.png', bbox_inches='tight')
    bar_plot(extractors, base_title, 'title')
    plt.savefig('base_title.png', bbox_inches='tight')

    names = ['sa', 'minfo', 'anova', 'pca']
    ensemble_text = {
        'kaggle': [.988, .987, .987, .983],
        'esp fake': [.851, .833, .831, .788],
        'mixed': [.913, .894, .903, .785],
    }
    ensemble_title = {
        'kaggle': [.955, .963, .963, .964],
        'esp fake': [.730, .737, .732, .674],
        'mixed': [.885, .905, .879, .760],
    }

    bar_plot(names, ensemble_text, 'text', different_colors=True)
    plt.savefig('ensemble_text.png', bbox_inches='tight')
    bar_plot(names, ensemble_title, 'title', different_colors=True)
    plt.savefig('ensemble_title.png', bbox_inches='tight')

    multilang_text = {
        'kaggle': [.924, .945, .925, .944, None],
        'esp fake': [.771, .773, .783, None, .748],
        'mixed': [.801, .793, .795, .773, .791],
    }
    multilang_title = {
        'kaggle': [.914, .926, .901, .923, None],
        'esp fake': [.633, .682, .685, None, .627],
        'mixed': [.751, .756, .750, .749, .743],
    }

    bar_plot(extractors, multilang_text, 'text')
    plt.savefig('multilang_text.png', bbox_inches='tight')
    bar_plot(extractors, multilang_title, 'title')
    plt.savefig('multilang_title.png', bbox_inches='tight')


def bar_plot(extractors, resutls_dict, attribute, different_colors=False):
    fig = plt.figure()

    fig.set_figheight(5)
    fig.set_figwidth(8)

    bar_width = 0.05
    dataset_labels_positions = []

    colors = sns.color_palette("husl", 9)
    if different_colors:
        colors = colors[5:]
    color_mapping = {e: c for e, c in zip(extractors, colors)}

    with sns.axes_style("darkgrid"):
        current_pos = 0.0
        for dataset, accuracy_list in resutls_dict.items():
            bars = []
            begin_pos = current_pos
            for extractor_name, acc in zip(extractors, accuracy_list):
                if acc == None:
                    continue
                bar = plt.bar(current_pos, acc, width=bar_width, align='edge', color=color_mapping[extractor_name])
                current_pos += bar_width
                bars.append(bar)

            dataset_pos = (current_pos + begin_pos) / 2
            dataset_labels_positions.append(dataset_pos)
            current_pos += 0.3

    plt.legend(bars, extractors, bbox_to_anchor=(1.15, 1), loc='upper right', borderaxespad=0)
    plt.grid(False)
    # plt.xlim(-0.1, 3.2)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.xticks(dataset_labels_positions, ['kaggle', 'esp fake', 'mixed'])
    plt.title(attribute)


if __name__ == '__main__':
    main()
