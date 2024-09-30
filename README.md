# Picture-Ranking-System

## This program is a Python script designed to analyze images using pretrained neural networks and hypergraphs. It processes the images to extract features, measures similarities between them, and creates a hypergraph to represent those relationships.
Here is a breakdown of what it does:
### 1) Extract Image Features:
The program uses pretrained models like ResNet50, GoogLeNet, or SqueezeNet to extract feature vectors from a set of images. These features represent the content of each image in a way that can be used to compare them.
### 2) Measure Hyperedge Similarity:
After extracting the features, it calculates the cosine similarity between the feature vectors of each pair of images. This similarity score measures how closely related two images are.
### 3) Create Hypergraph:
Based on the calculated similarities, the program creates a hypergraph. In this hypergraph, multiple images can be connected in a single "hyperedge" if their similarity exceeds a certain threshold (0.75 by default). This allows for a more complex representation than a traditional graph, which only connects two images at a time.
### 4) Adjust Ranking:
The program also normalizes the similarity scores between images to adjust their ranking. This ensures the scores are consistent and fall within a specific range.
### 5) Calculate Similarity Matrix:
Finally, a similarity matrix is created. This matrix contains similarity scores for all pairs of images that are part of the same hypergraph edges.
### 6) Cartesian Product of Hyperedges:
It calculates the Cartesian product of the hyperedges, essentially combining the images that are part of the same hypergraph edge for further analysis.
### 7) Output:
_The program prints:_
The extracted features of each image.
The similarities between images.
A similarity matrix showing how related each image is to others.
## Main Execution:
The main function (run_main) processes a list of image files (1.jpg, 2.jpg, etc.), extracts their features using the chosen model (ResNet50 by default), calculates their similarities, creates the hypergraph, and prints the similarity matrix.
In summary, the program analyzes images by creating a hypergraph based on the similarity of their features, allowing for advanced comparison and ranking of those images.
