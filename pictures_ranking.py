import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
import itertools

#  Ρύθμιση Κατάταξης (Ranking Adjustment) 
def adjust_ranking(ranks):
    normalized = (ranks - np.min(ranks)) / (np.max(ranks) - np.min(ranks))
    return normalized

#  Δημιουργία Υπεργράφου (Hypergraph Formation) 
def create_hypergraph(features, threshold_value):  
    edges = []
    for index1, feature_vector1 in enumerate(features):
        edge = [index1]
        for index2, feature_vector2 in enumerate(features):
            if index1 != index2 and measure_hyperedge_similarity(feature_vector1, feature_vector2) > threshold_value:
                edge.append(index2)
        if len(edge) > 1:
            edges.append(edge)
    print(f"Δημιουργημένες Υπερακμές: {edges}")  
    return edges

# Υπολογισμός Ομοιότητας Υπερακμών (Hyperedge Similarity Calculation) 
def measure_hyperedge_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)  

#  Υπολογισμός Καρτεσιανού Γινομένου Υπερακμών 
def cartesian_product(edges):
    return list(itertools.product(*edges))

#  Υπολογισμός Ομοιότητας Βάσει Υπεργράφου 
def calculate_hypergraph_similarity(edges, features, threshold_value):
    sim_matrix = np.zeros((len(features), len(features)))
    
    for edge in edges:
        for i, j in itertools.combinations(edge, 2):
            sim = measure_hyperedge_similarity(features[i], features[j])
            if sim >= threshold_value:
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim_matrix[i][j]

    return sim_matrix

# Εξαγωγή Χαρακτηριστικών από Εικόνες 
def extract_image_features(image_files, model_type):
    if model_type == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_type == 'googlenet':
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    elif model_type == 'squeezenet':
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
    else:
        raise ValueError("Μη έγκυρο μοντέλο")

    model.eval()  
    feature_list = []

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for path in image_files:
        img = Image.open(path)
        img_tensor = preprocess(img)
        batch_tensor = torch.unsqueeze(img_tensor, 0)

        with torch.no_grad():
            features = model(batch_tensor)
        feature_list.append(features.flatten().numpy())

    return feature_list

#  Κύρια Συναρτήση Εκτέλεσης 
def run_main():
    image_files = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
        "image4.jpg",
        "image5.jpg",
        "image6.jpg",
        "image7.jpg",
        "image8.jpg"
    ]  
    model_type = 'resnet50'  
    threshold_value = 0.6

    features = extract_image_features(image_files, model_type=model_type)

    print("Εξαγωγή Χαρακτηριστικών Εικόνων:")
    for idx, feature in enumerate(features):
        print(f"Εικόνα {idx}: {feature[:10]} ...")  

    # Υπολογισμός ομοιοτήτων μεταξύ εικόνων
    similarities = np.array([measure_hyperedge_similarity(v1, v2) for v1 in features for v2 in features])
    normalized_similarities = adjust_ranking(similarities)

    # Δημιουργία Υπεργράφου με τις κανονικοποιημένες ομοιότητες
    edges = create_hypergraph(features, threshold_value)

    print("\nΟμοιότητες Εικόνων:")
    for idx1, v1 in enumerate(features):
        for idx2, v2 in enumerate(features):
            sim = measure_hyperedge_similarity(v1, v2)
            print(f"Ομοιότητα μεταξύ εικόνας {idx1} και εικόνας {idx2}: {sim:.4f}")

    sim_matrix = calculate_hypergraph_similarity(edges, features, threshold_value)

    print("\nΜήτρα Ομοιότητας:")
    print(np.round(sim_matrix, 3))

# Εκκίνηση του Προγράμματος 

run_main()
