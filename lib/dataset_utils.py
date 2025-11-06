"""
Dataset utility functions for PartNet Mobility dataset.

This module provides utilities for working with the PartNet Mobility dataset,
including reading metadata and organizing objects by class.
"""

import os
import json
from typing import Dict, List


def get_dataset_dict(dataset_path: str = '/work/u9497859/shared_data/partnet-mobility-v0/dataset/') -> Dict[str, List[str]]:
    """
    Read the PartNet Mobility dataset and organize object IDs by class name.
    
    Args:
        dataset_path: Path to the root directory of the PartNet Mobility dataset.
                     Default is '/work/u9497859/shared_data/partnet-mobility-v0/dataset/'
    
    Returns:
        A dictionary mapping class names to lists of object IDs.
        Example: {'Door': ['1001', '1002'], 'Table': ['2001', '2002']}
    
    Raises:
        FileNotFoundError: If the dataset path doesn't exist.
        ValueError: If a meta.json file is missing or malformed.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Get all subdirectories (object IDs)
    folders = [name for name in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, name))]
    
    id_dict = {}
    
    for item in folders:
        meta_path = os.path.join(dataset_path, item, 'meta.json')
        
        # Check if meta.json exists
        if not os.path.exists(meta_path):
            print(f"Warning: meta.json not found for object {item}, skipping...")
            continue
        
        try:
            # Read the class from meta.json
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            _class = data.get('model_cat')
            
            if _class is None:
                print(f"Warning: 'model_cat' field missing in meta.json for object {item}, skipping...")
                continue
            
            # Add the object ID to the appropriate class
            if _class in id_dict:
                id_dict[_class].append(item)
            else:
                id_dict[_class] = [item]
        
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse meta.json for object {item}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing object {item}: {e}")
            continue
    
    # Sort object IDs within each class
    for cls in id_dict.keys():
        id_dict[cls].sort()
    
    return id_dict


def get_class_ids(class_name: str, dataset_path: str = '/work/u9497859/shared_data/partnet-mobility-v0/dataset/') -> List[str]:
    """
    Get all object IDs for a specific class.
    
    Args:
        class_name: The class name to filter by (e.g., 'Door', 'Table')
        dataset_path: Path to the root directory of the PartNet Mobility dataset.
    
    Returns:
        A sorted list of object IDs belonging to the specified class.
    
    Raises:
        ValueError: If the class name doesn't exist in the dataset.
    """
    id_dict = get_dataset_dict(dataset_path)
    
    if class_name not in id_dict:
        available_classes = sorted(id_dict.keys())
        raise ValueError(
            f"Class '{class_name}' not found in dataset. "
            f"Available classes: {', '.join(available_classes)}"
        )
    
    return id_dict[class_name]


def get_available_classes(dataset_path: str = '/work/u9497859/shared_data/partnet-mobility-v0/dataset/') -> List[str]:
    """
    Get a list of all available classes in the dataset.
    
    Args:
        dataset_path: Path to the root directory of the PartNet Mobility dataset.
    
    Returns:
        A sorted list of class names.
    """
    id_dict = get_dataset_dict(dataset_path)
    return sorted(id_dict.keys())
