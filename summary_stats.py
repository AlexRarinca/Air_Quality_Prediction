import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import random

# Load the dataset
data = np.load('image_data.npz')

train_x = data['train_X']
train_y = data['train_Y']
test_x = data['test_X']
test_y = data['test_Y']

# Descriptive Statistics -------------------------------------------
def descriptive_stats(data):
    CT = [np.mean(data), np.median(data), stats.mode(data)]
    V = [np.var(data), np.std(data), np.percentile(data, 75) - np.percentile(data, 25)]
    F = [np.histogram(data, bins=10)]
    return CT, V, F

train_x_CT, train_x_V, train_x_F = descriptive_stats(train_x)
train_y_CT, train_y_V, train_y_F = descriptive_stats(train_y)
test_x_CT, test_x_V, test_x_F = descriptive_stats(test_x)
test_y_CT, test_y_V, test_y_F = descriptive_stats(test_y)

# Retreive the number of classes and their populations
classes, count  = np.unique(train_y, return_counts=True)

plt.figure(figsize=(10, 5))
plt.bar(classes, count)
plt.title('Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Population')
plt.show()

chosen_classes = [rand_classes[1], rand_classes[3]]
class1_vis = []
class2_vis = []

for i in range(0,train_y.size):
    if train_y[i] == chosen_classes[0]:
        class1_vis.append(train_x[i])
    elif train_y[i] == chosen_classes[1]:
        class2_vis.append(train_x[i])

# Convert lists to NumPy arrays
class1_vis = np.array(class1_vis)
class2_vis = np.array(class2_vis)
combined_vis = np.vstack((class1_vis, class2_vis))
# Reduce dimensionality to 2D using PCA for visualization
pca = PCA(n_components=2)
combined_vis_pca = pca.fit_transform(combined_vis)
# Split the PCA-transformed data back into classes
class1_vis_pca = combined_vis_pca[:len(class1_vis)]
class2_vis_pca = combined_vis_pca[len(class1_vis):]

# Scatter plot for class1_vis and class2_vis
plt.figure(figsize=(10, 5))
plt.scatter(class1_vis_pca[:, 0], class1_vis_pca[:, 1], color='blue', label='Class 1')
plt.scatter(class2_vis_pca[:, 0], class2_vis_pca[:, 1], color='red', label='Class 2')
plt.title('Scatter Plot of Class 1 and Class 2 in train_X')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# Choose class 0 for first subset
subset1 = []
subset2 = []
for i in range(0,train_y.size):
    if train_y[i] == 0:
        subset1.append(train_x[i])

# Choose random classes for second subset
while len(subset2) < len(subset1):
    random_index = np.random.randint(0, classes.size)
    subset2.append(train_x[random_index])

subset1 = np.array(subset1)
subset2 = np.array(subset2)
def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Function to compute pairwise MSE for a subset
def pairwise_mse(subset):
    mse_scores = []
    num_images = subset.shape[0]
    for i in range(num_images):
        for j in range(i+1, num_images):
            mse = calculate_mse(subset[i], subset[j])
            mse_scores.append(mse)
    return mse_scores

# Compute MSE scores for both subsets
mse_subset1 = pairwise_mse(subset1)
mse_subset2 = pairwise_mse(subset2)


plt.figure(figsize=(12, 6))

# Histogram for Subset1
plt.hist(mse_subset1, bins=30, alpha=0.5, label='Subset1 (Class "0")', color='blue')

# Histogram for Subset2
plt.hist(mse_subset2, bins=30, alpha=0.5, label='Subset2 (Random)', color='red')

plt.title('Histogram of MSE Similarity Scores')
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
