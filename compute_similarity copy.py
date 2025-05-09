import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import os

def compute_mean_pr(feats, labels):
    n = len(labels)
    norm_feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    sim = norm_feats @ norm_feats.T

    all_precisions = []
    recall_grid = np.linspace(0, 1, 100)

    for i in range(n):
        query_label = labels[i]
        y_true = np.array([1 if labels[j] == query_label and j != i else 0 for j in range(n)])
        y_score = sim[i]
        rec, prec = precision_recall_curve(y_true, y_score)
        # interpolamos para comparar en puntos fijos
        prec_interp = np.interp(recall_grid, rec, prec, left=1.0, right=0.0)
        all_precisions.append(prec_interp)
    
    mean_precision = np.mean(all_precisions, axis=0)

    # Graficar
    plt.figure(figsize=(6,4))
    plt.plot(recall_grid, mean_precision, label=f'{MODEL.upper()}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {MODEL.upper()} on {DATASET}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def precision_recall_curve(y_true, y_score):
    """Calcula la curva P-R para una query"""
    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]
    
    precisions = []
    recalls = []
    tp = 0
    total_positives = np.sum(y_true)
    
    for i, label in enumerate(y_true_sorted):
        if label:
            tp += 1
        precision = tp / (i + 1)
        recall = tp / total_positives if total_positives else 0
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(recalls), np.array(precisions)


def average_precision(y_true, y_score):
    """Calcula el average precision de una consulta"""
    sorted_indices = np.argsort(-y_score)  # ordenar de mayor a menor score
    y_true_sorted = y_true[sorted_indices]
    
    precisions = []
    num_relevant = 0
    for i, relevant in enumerate(y_true_sorted):
        if relevant:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    if not precisions:
        return 0.0
    return np.mean(precisions)

def compute_map(feats, labels):
    n = len(labels)
    norm_feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    sim = norm_feats @ norm_feats.T
    mAPs = []
    for i in range(n):
        query_label = labels[i]
        # excluimos la query de sí misma
        y_true = np.array([1 if labels[j] == query_label and j != i else 0 for j in range(n)])
        y_score = sim[i]
        mAPs.append(average_precision(y_true, y_score))
    return np.mean(mAPs), mAPs


# load the data for visualizing the results
data_dir = 'Paris'
image_dir = os.path.join(data_dir, 'images')
val_file = os.path.join(data_dir, 'list_of_images.txt')

DATASET = 'paris'
MODEL = 'clip'  # Changed to use CLIP by default
feat_file = os.path.join('data', 'feat_{}_{}.npy'.format(MODEL, DATASET))
if __name__ == '__main__' :
    with open(val_file, "r+") as file: 
        files = [f.split('\t') for f in file]

    feats = np.load(feat_file)    
    norm2 = np.linalg.norm(feats, ord = 2, axis = 1,  keepdims = True)
    feats_n = feats / norm2
    sim = feats_n @ np.transpose(feats_n)
    sim_idx = np.argsort(-sim, axis = 1)

    # 🔽 AÑADE ESTO
    labels = [f[1].strip() for f in files]
    map_score, ap_list = compute_map(feats, labels)
    print(f"\n📊 Mean Average Precision (mAP): {map_score:.4f}")
    
    compute_mean_pr(feats, labels)

    # Obtener los índices de las 5 peores queries
    worst_queries = np.argsort(ap_list)[:5]
    print("🟥 5 queries con peor AP:", worst_queries)
    
    best_queries = np.argsort(ap_list)[-5:][::-1]  # orden descendente
    print("🟩 5 queries con mejor AP:", best_queries)


    # Visualización ejemplo
    query = np.random.permutation(sim.shape[0])[0]
    k = 10
    best_idx = sim_idx[query, :k+1]
    print(sim[query, best_idx])

    fig, ax = plt.subplots(1,11)
    for i, idx in enumerate(best_idx):        
        filename = os.path.join(image_dir, files[idx][0])
        im = io.imread(filename)
        im = transform.resize(im, (64,64)) 
        ax[i].imshow(im)                 
        ax[i].set_axis_off()
        ax[i].set_title(files[idx][1])
            
    ax[0].patch.set(lw=6, ec='b')
    ax[0].set_axis_on()            
    plt.show()
    
    # Mostrar recuperación de los 5 peores
    for i, q_idx in enumerate(worst_queries):
        print(f"\n🔍 Worst Query #{i+1} - AP: {ap_list[q_idx]:.4f}")
        best_idx = sim_idx[q_idx, :11]  # query + 10 más similares
        fig, ax = plt.subplots(1, 11, figsize=(12, 2))
        for j, idx in enumerate(best_idx):
            filename = os.path.join(image_dir, files[idx][0])
            im = io.imread(filename)
            im = transform.resize(im, (64,64)) 
            ax[j].imshow(im)
            ax[j].set_axis_off()
            ax[j].set_title(files[idx][1])
        ax[0].patch.set(lw=6, ec='r')
        ax[0].set_axis_on()
        plt.suptitle(f"Query #{i+1} - Clase: {files[q_idx][1].strip()} - AP: {ap_list[q_idx]:.4f}")
        plt.show()
    
    # Mostrar recuperación de los 5 mejores
    for i, q_idx in enumerate(best_queries):
        print(f"\n🏆 Best Query #{i+1} - AP: {ap_list[q_idx]:.4f}")
        best_idx = sim_idx[q_idx, :11]  # query + 10 más similares
        fig, ax = plt.subplots(1, 11, figsize=(12, 2))
        for j, idx in enumerate(best_idx):
            filename = os.path.join(image_dir, files[idx][0])
            im = io.imread(filename)
            im = transform.resize(im, (64,64)) 
            ax[j].imshow(im)
            ax[j].set_axis_off()
            ax[j].set_title(files[idx][1])
        ax[0].patch.set(lw=6, ec='g')  # verde para mejor
        ax[0].set_axis_on()
        plt.suptitle(f"Best Query #{i+1} - Clase: {files[q_idx][1].strip()} - AP: {ap_list[q_idx]:.4f}")
        plt.show()

        
    

