import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_avrg_features(train_dataset_name, dataset_name, extraction_method, attribute):
    fold_idx = 0
    features = np.load(f'extracted_features/{train_dataset_name}_{extraction_method}_{dataset_name}/fold_{fold_idx}_X_train_{attribute}.npy')
    avrg_features = np.mean(features, axis=0)
    return avrg_features


def main():
    methods = ('lda', 'tf_idf')
    attribute = 'text'
    fig, axs = plt.subplots(3 * len(methods), 3)

    for i, extraction_method in enumerate(methods):
        for j, train_dataset_name in enumerate(('esp_fake', 'bs_detector', 'mixed')):
            for k, dataset_name in enumerate(('esp_fake', 'bs_detector', 'mixed')):
                avrg_features = get_avrg_features(train_dataset_name, dataset_name, extraction_method, attribute)
                axs[i * 3 + j][k].get_yaxis().set_ticklabels([])
                axs[i * 3 + j][k].get_xaxis().set_ticklabels([])
                axs[i * 3 + j][k].grid(False)
                sns.heatmap(np.expand_dims(avrg_features, 0), xticklabels=False, yticklabels=False, cbar=False, ax=axs[i * 3 + j][k])
                if k == 0:
                    axs[i * 3 + j][0].set_ylabel(train_dataset_name)
                if j == 0 and i == 0:
                    axs[i * 3 + j][k].xaxis.set_label_position('top') 
                    axs[i * 3 + j][k].set_xlabel(dataset_name)
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()