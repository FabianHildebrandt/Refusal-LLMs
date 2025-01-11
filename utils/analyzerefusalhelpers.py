# %%
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .gdv import cmpGDV
import pandas as pd
import os 
import seaborn as sns
from sklearn.manifold import TSNE
import umap  # Make sure you have umap-learn installed (pip install umap-learn)
import pickle
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# %%
def save_model_fig(model_name : str, f_name : str ):
    os.makedirs(model_name, exist_ok=True)
    fpath = os.path.join(model_name, f_name)
    plt.savefig(fpath)
    print(f'Successfully stored file under the following path {fpath}')

# %%
def dim_reduction_pca(data, n_dim : int = 2):
    """Function to scale, center and perform PCA on data
    
    Args:
        data (array-like): data to perform PCA
        n_dim (int): Number of dimensions to keep 

    Returns:
        data_pca (ndarray): dimensionality-reduced data 
    
    """

    # print(f'Processing data with shape: {data.shape[0]} x {data.shape[1]}')

    scaler = StandardScaler()

    data = scaler.fit_transform(data)
    
    pca = PCA(n_dim)
    transformed_data = pca.fit_transform(data)

    # print(f'Successfully performed PCA on data, resulting shape: {transformed_data.shape[0]} x {transformed_data.shape[1]}')
    return transformed_data


def dim_reduction_tsne(data, n_dim: int = 2, random_state: int = 42):
    """Function to perform t-SNE dimensionality reduction
    Args:
        data (array-like): data to perform t-SNE
        n_dim (int): Number of dimensions to keep
        random_state (int): Random seed for reproducibility
    Returns:
        data_tsne (ndarray): dimensionality-reduced data
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    tsne = TSNE(
        n_components=n_dim, 
        random_state=random_state, 
        max_iter=1000,  # Increased iterations for better convergence
        perplexity=min(30, (data.shape[0] - 1) // 3)  # Adaptive perplexity
    )
    
    transformed_data = tsne.fit_transform(data)
    
    return transformed_data

def dim_reduction_umap(data, n_dim: int = 2, random_state: int = 42):
    """Function to perform UMAP dimensionality reduction
    Args:
        data (array-like): data to perform UMAP
        n_dim (int): Number of dimensions to keep
        random_state (int): Random seed for reproducibility
    Returns:
        data_umap (ndarray): dimensionality-reduced data
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    reducer = umap.UMAP(
        n_components=n_dim, 
        #random_state=random_state,
        n_neighbors=15,  # Default neighborhood size
        min_dist=0.1,    # Balance between preserving local and global structure
        metric='euclidean'  # Distance metric
    )
    
    transformed_data = reducer.fit_transform(data)
    
    return transformed_data

# %%
def flip_components(data: dict, label1_size: int) -> dict:
    """
    Ensures that the component directions of subsequent layers
    face the same direction for consistency by comparing the difference vector
    between the two label means across iterations.

    Args:
        data (dict): Dictionary containing dimensionality-reduced data for multiple layers.
                    Each layer should have the key "data" with a numpy array of shape (n_samples, n_components).
        label1_size (int): Number of samples belonging to label1 in each layer.

    Returns:
        dict: Data dictionary with corrected component directions.
    """
    # Iterate through all layers
    for layer in data.keys():
        # Extract dimensionality-reduced data for the current layer
        layer_data = data[layer]["data"]

        # Compute the mean of components for label1 and label2
        label1_comp_mean = layer_data[:label1_size, :].mean(axis=0)
        label2_comp_mean = layer_data[label1_size:, :].mean(axis=0)

        # Compute the difference vector
        diff_vector = label2_comp_mean - label1_comp_mean

        # Check the sign of the difference vector
        sign_vector = np.sign(diff_vector)

        # Flip the components if needed
        layer_data *= sign_vector

        # Update the data dictionary with the modified layer data
        data[layer]["data"] = layer_data

    return data

# %%
def calcualte_dim_reduced_data(data : dict, label1 : str, label2 :str, method : str ):
    """
    Function to compute the dim.-reduced activations, GDV, inter- and intra-class distance

    Args:
        data (dict) : dictionary containing the high-dimensional hidden activations 
        label1 (str) : Name of the first set of activations
        label2 (str) : Name of the second set of activations
        method (str) : name of the dimensionality-reduction method that should be used
    """
    data_label1_shape = list(data[label1].values())[0].shape
    data_label2_shape = list(data[label2].values())[0].shape

    print(f'Data shape {label1}: {len(data[label1])} x {data_label1_shape[0]} x {data_label1_shape[1]}')
    print(f'Data shape {label2}: {len(data[label2])} x {data_label2_shape[0]} x {data_label2_shape[1]}')

    dim_reduced_data = {}

    for layer in data[label1].keys():
        if layer not in data[label2].keys():
            print(f'The layer {layer} from {label1} can\'t be found in the dataset {label2}')
            continue
        # pca_layer_data = np.zeros_like()
        cat_data = torch.cat((data[label1][layer].to(torch.float32),data[label2][layer].to(torch.float32)), dim=0)

        if method.lower() == 'pca':
            dim_reduced_layer_data = dim_reduction_pca(cat_data)
        elif method.lower() == 't-sne':
            dim_reduced_layer_data = dim_reduction_tsne(cat_data)
        elif method.lower() == 'umap':
            dim_reduced_layer_data = dim_reduction_umap(cat_data)
        else:
            raise ValueError(f'The dimensionality-reduction method {method} is not supported.')
        
        label1_size = data[label1][layer].shape[0]
        label2_size = data[label2][layer].shape[0]
        
        labels = label1_size * [0] + label2_size * [1]

        intra, inter, gdv = cmpGDV(dim_reduced_layer_data, labels) 
        dim_reduced_data[layer] = {'data':dim_reduced_layer_data, 'gdv':gdv, 'intra':intra, 'inter': inter}

    return dim_reduced_data


# %%
def analyze_gdv(data : np.ndarray, model_name : str, method : str, method_hyperparams : str = ""):
    """Analyze the GDV of the data across all layers and store results in a dataframe"""

    gdv_info = []
    for layer in data.keys():
        gdv_info.append( {'layer_name': layer, 'gdv':data[layer]['gdv'], 'intra':data[layer]['intra'], 'inter': data[layer]['inter']})

    gdv_info = pd.DataFrame(gdv_info)
    gdv_info = gdv_info.set_index('layer_name')
    gdv_info_normalized = gdv_info.copy()
    gdv_info_normalized['gdv'] = 100 * gdv_info_normalized['gdv'] / gdv_info_normalized['gdv'].min()
    gdv_info_normalized['intra'] = 100 * gdv_info_normalized['intra'] / gdv_info_normalized['intra'].max()
    gdv_info_normalized['inter'] = 100 * gdv_info_normalized['inter'] / gdv_info_normalized['inter'].max()

    min_gdv = gdv_info['gdv'].min()
    min_gdv_layer = gdv_info[gdv_info['gdv'] == min_gdv].index[0]
    min_intra = gdv_info['intra'].min()
    max_inter = gdv_info['inter'].max()
    num_layers = len(data.keys()) / 2
    gdv_summary = {'Model': model_name, 'Method': method, 'Method Hyperparameters' : method_hyperparams, 'GDV':min_gdv, 'Layer': min_gdv_layer, 'Minimum Intra-class':min_intra, 'Maximum Inter-class':max_inter, 'Model Layers':num_layers}

    return gdv_info, gdv_info_normalized, gdv_summary


# %%
def plot_gdv_across_layers(gdv_normalized : pd.DataFrame, model_name : str, label1 : str, label2 :str, method : str, method_hyperparams : str = ""):
    """Creates a heatmap for the GDV activations across the layers """

    # Create a figure with appropriate size
    plt.figure(figsize=(18, 5))

    plt.title(f'{model_name} - Separability Analysis of {label1} and {label2} instructions')

    # Create heatmap
    sns.heatmap(gdv_normalized.T,  # Transpose to get layers as columns
                cmap='RdBu_r',  # Red-Blue diverging colormap (reversed to make red=1, blue=0)
                center=50,     # Center the colormap at 0.5
                vmin=0,         # Minimum value
                vmax=100,         # Maximum value
                annot=False,     # Show values in cells
                cbar_kws={'label': 'Percentage of Max Value [%]'},
                xticklabels=True,
                yticklabels=['GDV', 'Intra-class', 'Inter-class'])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if method_hyperparams == "":
        f_name = f'GDV_{method}.png'
    else:
        f_name = f'GDV_{method}-{method_hyperparams}.png'
    save_model_fig(model_name, f_name)

# %%
def plot_all_layers(data: np.ndarray, gdv : pd.DataFrame, model_name: str, label1: str, label2: str, label1_size, method : str, method_hyperparams : str = ""):
    """
    Function receives dimensionality-reduced data and visualizes the activations for each layer.
    To simplify the activations, the mean and covariance for both labels are determined and plotted.
    
    Args:
        data (dict): Dictionary containing layer-wise dimensionality-reduced data
        model_name (str): name of the used model
        label1 (str): Label 1 displayed in red
        label2 (str): Label 2 displayed in blue
        label1_size (int): Size of the first label group
        method (str) : method used for dimensionality reduction
        method_hyperparams (str) : method hyperparameters (e. g. perplexity for t-SNE)
    """
    num_layers = len(data.keys())
    num_columns = 8
    num_rows = (num_layers + num_columns - 1) // num_columns  # Ceiling division

    # depending on the method determine the used x and y labels
    if method.lower() == "pca":
        xlabel = 'PC 1'
        ylabel = 'PC 2'
    elif method.lower() == "umap":
        xlabel = 'UMAP 1'
        ylabel = 'UMAP 2'
    else:
        xlabel = 'Dim 1'
        ylabel = 'Dim 2'
    
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 3, num_rows * 3))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    if method_hyperparams == "":
        method_suptitle = method
    else:
        method_suptitle = f'{method} ({method_hyperparams})'

    fig.suptitle(f'{model_name} - {method_suptitle} across the model layers', y = 0.99)
    
    plot_idx = 0
    for layer in data.keys():

        # Calculate row and column indices
        row_idx = plot_idx // num_columns
        col_idx = plot_idx % num_columns
        
        # Access the correct subplot
        if num_rows == 1:
            ax = axs[col_idx]
        else:
            ax = axs[row_idx, col_idx]
            
        # Create scatter plots
        ax.scatter(data[layer]['data'][:label1_size, 0], 
                  data[layer]['data'][:label1_size, 1], 
                  color='r', 
                  label=label1,
                  alpha=0.6)
        
        ax.scatter(data[layer]['data'][label1_size:, 0], 
                  data[layer]['data'][label1_size:, 1], 
                  color='b', 
                  label=label2,
                  alpha=0.6)
        
        # Set title and labels
        ax.set_title(f'{layer} | GDV: {gdv.loc[layer, 'gdv']:.2f}', pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend if it's the first plot
        if plot_idx == 0:
            ax.legend(loc= 'upper right')
            
        plot_idx += 1
    
    # Hide empty subplots
    for idx in range(plot_idx, num_rows * num_columns):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        if num_rows == 1:
            axs[col_idx].set_visible(False)
        else:
            axs[row_idx, col_idx].set_visible(False)
    
    plt.tight_layout()

    save_model_fig(model_name, f'Scatter_{method_suptitle}.png')

    return fig, axs

# %%
def plot_single_layer(data: np.ndarray, layer : str, gdv : float, model_name: str, label1: str, label2: str, label1_size, method : str, method_hyperparams : str = ""):
    """
    Creates a scatter plot of a specific layer, that is passed as an argument

    Args:
        data (dict): Dictionary containing layer-wise dimensionality-reduced data
        model_name (str): name of the used model
        label1 (str): Label 1 displayed in red
        label2 (str): Label 2 displayed in blue
        label1_size (int): Size of the first label group
        method (str) : method used for dimensionality reduction
        method_hyperparams (str) : method hyperparameters (e. g. perplexity for t-SNE)
    """
    # depending on the method determine the used x and y labels
    if method.lower() == "pca":
        xlabel = 'PC 1'
        ylabel = 'PC 2'
    elif method.lower() == "umap":
        xlabel = 'UMAP 1'
        ylabel = 'UMAP 2'
    else:
        xlabel = 'Dim 1'
        ylabel = 'Dim 2'
    
    if method_hyperparams == "":
        method_suptitle = method
    else:
        method_suptitle = f'{method} ({method_hyperparams})'


    # create the figure
    plt.figure(figsize=(3.5,3.5))
    plt.title(f'{model_name} - {method_suptitle}\nLayer: {layer} | GDV: {gdv:.2f}', fontweight='bold', fontsize=10)
            
    # Create scatter plots
    plt.scatter(data[layer]['data'][:label1_size, 0], 
                data[layer]['data'][:label1_size, 1], 
                color='r', 
                label=label1,
                alpha=0.6)
    
    plt.scatter(data[layer]['data'][label1_size:, 0], 
                data[layer]['data'][label1_size:, 1], 
                color='b', 
                label=label2,
                alpha=0.6)
        
    # Set labels
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)   
    # plt.legend(loc= 'upper right')
    plt.legend()
    
    plt.tight_layout()

    plt.subplots_adjust(top=0.85)

    save_model_fig(model_name, f'Scatter_Single_{method_suptitle}.png')


# %%
def get_axis_ranges(data : dict):
    '''Function expects data of LLM layers (each key is a layer name, value is the corresponding data) and returns axis ranges for each dimension of the data.
    
    Args:
        data (dict): key - layer name, value - layer data

    Returns:
        ranges (ndarray): dimensionality 2 (min, max) x n_dim_data
    
    '''

    print('Identifying the axis ranges of the data...')

    n_dim = data[list(data.keys())[0]]["data"].shape[1]
    ranges = np.zeros((2, n_dim))
    ranges[0,:] = np.inf
    ranges[1,:] = -np.inf

    for layer in data.keys():
        layer_max = data[layer]["data"].max(axis=0)
        layer_min = data[layer]["data"].min(axis=0)
        ranges[0] = np.minimum(ranges[0], layer_min)
        ranges[1] = np.maximum(ranges[1], layer_max)
    
    return ranges

# %%
def animate_layer_activations(data: dict, gdv: pd.DataFrame, model_name: str, 
                               label1: str, label2: str, label1_size: int, 
                               method: str, method_hyperparams: str = ""):
    """
    Create an animation of layer-wise scatter plots with each frame visible for 0.2 seconds.
    
    Args:
    data (dict): Dictionary containing layer-wise dimensionality-reduced data
    gdv (pd.DataFrame): Dataframe with Gradient Dissimilarity Values
    model_name (str): Name of the used model
    label1 (str): Label 1 displayed in red
    label2 (str): Label 2 displayed in blue
    label1_size (int): Size of the first label group
    method (str): Method used for dimensionality reduction
    method_hyperparams (str, optional): Method hyperparameters 
    output_filename (str, optional): Name of the output MP4 file
    
    Returns:
    matplotlib.animation.FuncAnimation: The created animation
    """
    # Determine x and y labels based on dimensionality reduction method
    if method.lower() == "pca":
        xlabel = 'PC 1'
        ylabel = 'PC 2'
    elif method.lower() == "umap":
        xlabel = 'UMAP 1'
        ylabel = 'UMAP 2'
    else:
        xlabel = 'Dim 1'
        ylabel = 'Dim 2'
    
    # Prepare the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(top=0.9)
    
    # If method hyperparameters are provided, include them in the title
    if method_hyperparams:
        method_suptitle = f'{method} ({method_hyperparams})'
    else:
        method_suptitle = method

    # Initialize the scatter plots with the first layer's data
    scatter1 = ax.scatter([], [], color='r', label=label1, alpha=0.6)
    scatter2 = ax.scatter([], [], color='b', label=label2, alpha=0.6)

    ranges = get_axis_ranges(data)
    ranges *= 1.5

    ax.set_xlim(ranges[0,0], ranges[1,0])  
    ax.set_ylim(ranges[0,1], ranges[1,1])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc='upper right')

    # Update function for the animation
    def update(frame):
        layer_name = list(data.keys())[frame]
        layer_data = data[layer_name]['data']
        
        # Update scatter data
        scatter1.set_offsets(layer_data[:label1_size, :2])
        scatter2.set_offsets(layer_data[label1_size:, :2])
        
        # Update title
        ax.set_title(f'{model_name} - {method_suptitle}\n{layer_name}', fontsize=16)
        return scatter1, scatter2

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(data.keys()), interval=200, blit=True)

    output_filename=f'Scatter_Animation_{method_suptitle}.mp4'
    os.makedirs(model_name, exist_ok=True)
    fpath = os.path.join(model_name, output_filename)
    ani.save(fpath, writer='ffmpeg', fps=5)
    print(f'Successfully stored file under the following path {fpath}')
    
    plt.close(fig)
    
    return ani


# %%
def store_model_embeddings(model_embeddings : dict, model_name : str, labels : list, label_sizes : list):
    ''' 
    Function that stores the model embeddings along with the labels in a pickle file

    Pickle file has the following contents
        data - dimensionalit-reduced data of the different methods
        labels - list of tuples [(label1, labe1_size), (label2, label2_size), ...]
    '''

    data = {
        'data': model_embeddings,
        'labels': zip(labels, label_sizes)
    }

    os.makedirs(model_name, exist_ok=True)
    f_name = 'model_embeddings.pkl'
    fpath = os.path.join(model_name, f_name)
    
    print(data['labels'])
    with open(fpath, 'wb') as file:
        pickle.dump(data, file)

    print(f'Successfully stored the model embeddings for the model {model_name} in the under the filepath {fpath}.')
    

# %%
def calculate_refusal_vector(data, label1_size):
    '''
    Returns an n-dim vector pointing towards refusal using difference-in-means
    '''

    label1_data = data[:label1_size][:]
    label2_data = data[label1_size:][:]

    label1_mean = label1_data.mean(axis=0)
    label2_mean = label2_data.mean(axis=0)

    return label1_mean - label2_mean

# %%
def rank_instructions_layer_wise(data, label1_size):
    '''
    Returns a dictionary containing the refusal vectors for the different layers of a model
    '''

    refusal_rankings = {}
    # iterate through all model layers
    for model_layer in data.keys():
        layer_embeddings = data[model_layer]['data']
        # calculate the refusal direction for the current layer and append it to the dictionary
        layer_refusal_vector = calculate_refusal_vector(layer_embeddings, label1_size).reshape(-1,1)
        # dot product between embeddings and the refusal vector
        refusal_scores = np.dot(layer_embeddings, layer_refusal_vector).flatten()
        # sort the instructions based on the refusal 
        refusal_rankings[model_layer] = np.argsort(refusal_scores).flatten()

    return refusal_rankings

# %%
def exchange_dict_keys(original_dict, old_keys, new_keys):
    # Create a new dictionary with swapped keys
    new_dict = dict(zip(new_keys, [original_dict[key] for key in old_keys]))
    return new_dict

# %%
def run_pipeline(activations_path : str, model_name : str, methods : list, label1: str, label2 : str, gdv_overview : list, postprocessing : bool = True):
    """Run full data processing pipeline"""

    # load activations from the pickle file
    with open(activations_path, 'rb') as f:
        data = pickle.load(f)

    label1_size = list(data[label1].values())[0].shape[0]
    label2_size = list(data[label2].values())[0].shape[0]
    print('-------------')
    print(f'{model_name}'.upper())
    print('-------------')
    print(f'Dataset keys: {data.keys()}')
    print(f'Label distribution: {label1_size} for {label1}, {label2_size} for {label2}')


    new_layer_names = []
    remove_list = []
    for layer in data[label1].keys():
        layer_no = layer.split(sep='.')[1]
        if 'resid_mid' in layer:
            layer_type = 'attention'
        elif 'resid_post' in layer:
            layer_type = 'mlp'
        else:
            remove_list.append(layer)
            continue
        new_layer_names.append(f'{layer_no}_{layer_type}')

    for layer in remove_list:
        try:
            data[label1].pop(layer)
            data[label2].pop(layer)
        except:
            pass

    
    data[label1] = exchange_dict_keys(data[label1], list(data[label1].keys()), new_layer_names)
    data[label2] = exchange_dict_keys(data[label2], list(data[label2].keys()), new_layer_names)

    print(f'Layers: {list(data[label1].keys())}')

    model_embeddings = {}
    refusal_vectors = {}

    for idx, method in enumerate(methods):
        print(f'Performing dimensionality reduction using {method}...')
        processed_data = calcualte_dim_reduced_data(data, label1, label2, method)
        if postprocessing:
            print(f'Ensure consistent directions of the components...')
            processed_data = flip_components(processed_data, label1_size)
        print(f'Calculating the GDV for the dimensionality-reduced data of the method {method}...')
        gdv, gdv_normalized, gdv_summary = analyze_gdv(processed_data, model_name, method)
        print(f'Visualizing the dimensionality-reduced data of the method {method} for all layers...')
        plot_all_layers(processed_data, gdv, model_name, label1, label2, label1_size, method)
        print(f'Visualizing the dimensionality-reduced data of the method {method} for the most discriminating layer {gdv_summary["Layer"]}...')
        plot_single_layer(processed_data, layer=gdv_summary['Layer'], gdv=gdv_summary['GDV'], model_name=model_name, label1=label1, label2=label2, label1_size=label1_size, method=method)
        print(f'Animating the dimensionality-reduced data of the method {method}...')
        animate_layer_activations(processed_data, gdv, model_name, label1, label2, label1_size, method)
        print(f'Visualizing the GDV across the layers for the method {method}...')
        plot_gdv_across_layers(gdv_normalized,model_name, label1, label2, method)
        print(f'Calculating the refusal directions...')
        refusal_vectors[method] = rank_instructions_layer_wise(processed_data, label1_size)
        # add the dimensionality-reduced data of the current method to a dictionary
        model_embeddings[method] = processed_data

        gdv_overview.append(gdv_summary)

    print(f'Storing the model embeddings for model {model_name}...')
    store_model_embeddings(model_embeddings, model_name, [label1, label2], [label1_size, label2_size])
