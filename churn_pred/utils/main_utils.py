import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from box.exceptions import BoxValueError
from box import ConfigBox
from churn_pred import logger
import yaml
import json
from ensure import ensure_annotations
from pathlib import Path

@ensure_annotations
def create_directories(paths_to_directories: list, verbose: bool=True):
    """
    create list of directories

    Args:
        paths_to_directories (list of paths): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created.
                                     Defaults to False.
    """
    
    for path in paths_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.full_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    
    except Exception as e:
        raise e

@ensure_annotations
def get_size(path: Path) -> str:
    """
    get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    
    size_in_kb = round(os.path.getsize(path)/1024)
    
    return f"~ {size_in_kb} KB"

@ensure_annotations
def save_json(path, data, name):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    file_pth = os.path.join(path, f"{name}.json")
    
    with open(file_pth, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {file_pth}")

def plot_confusion_matrix(conf_matrix, cm_path, model_name):
    # Transpose the matrix to match y-axis as Predicted, x-axis as Actual
    TN = conf_matrix[0,0]
    FP = conf_matrix[0,1]
    FN = conf_matrix[1,0]
    TP = conf_matrix[1,1]
    
    conf_matrix = np.array([[TP, FP], [FN, TN]])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[1, 0], yticklabels=[1, 0], 
                cbar=True, annot_kws={"size": 12})
    
    # Set axis labels
    plt.xlabel("Actual", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    
    plt.title("Confusion Matrix for {model_name}")
    
    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(os.path.join(cm_path, f'cm_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
