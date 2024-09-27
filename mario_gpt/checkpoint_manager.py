"""
Author: Balasubramani Murugan

This script provides utility functions for saving, loading, and managing model checkpoints
in a PyTorch-based training pipeline. It includes functionality to track validation loss
across epochs and manage a limited number of saved checkpoints, while also keeping track
of the best model based on validation performance.
"""
import logging
import os
import json
import shutil
import torch

logger = logging.getLogger(__name__)


def get_checkpoint_details(checkpoint_dir, checkpoint_name) -> (int, float):
    """
    This function return particular checkpoint details such as epoch and validation loss

    @param checkpoint_dir: checkpoint_dir: Path of the directory in which checkpoint is stored
    @param checkpoint_name: checkpoint_name: Checkpoint name
    @return (int,float) checkpoint's epoch and validation loss
    """
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path)
    checkpoint_epoch = checkpoint['epoch']
    checkpoint_val_loss = checkpoint['val_loss']
    return checkpoint_epoch, checkpoint_val_loss


def save_model_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    """
    This function will store the given model and optimizer in the given checkpoint directory.

    @param model: Model that need to be saved
    @param optimizer: Optimizer state that need to be saved
    @param epoch: Current training epoch of the model
    @param val_loss: Current Validation loss of the model
    @param checkpoint_path: Path of the checkpoint directory
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")


def load_metadata(meta_file_path):
    """
    This function load the checkpoint.json file and return as dict

    @param meta_file_path: Path of the checkpoint meta file i.e. checkpoint.json
    @return: Return the meta file content as a dict
    """
    if os.path.exists(meta_file_path):
        with open(meta_file_path, 'r') as meta_file:
            return json.load(meta_file)
    else:
        return {'checkpoints': [], 'best_checkpoint': None}


def update_metadata(metadata, checkpoint_dir, checkpoint_name, epoch, val_loss, is_current_best=False):
    """
    This function is to update the metadata information with the recent epoch information along wil the managing
    the best model information.

    @param metadata: All checkpoint metadata in the dict type
    @param checkpoint_dir: Path of the checkpoint directory
    @param checkpoint_name: Checkpoint name
    @param epoch: Current training epoch of the model
    @param val_loss: Current Validation loss of the model
    @param is_current_best: Boolean flag whether is the current model is the best model or not
    """
    checkpoint_exists = False

    # Check if the checkpoint already exists in metadata
    for ckpt in metadata['checkpoints']:
        if ckpt['name'] == checkpoint_name:
            ckpt['epoch'] = epoch
            ckpt['val_loss'] = val_loss
            checkpoint_exists = True
            logger.info(f"Metadata updated for existing checkpoint: {checkpoint_name}")
            break

    if not checkpoint_exists:
        # Add new checkpoint metadata
        metadata['checkpoints'].append({
            'name': checkpoint_name,
            'epoch': epoch,
            'val_loss': val_loss
        })

    # Update the best checkpoint if needed
    if metadata.get('best_checkpoint') is None or is_current_best:
        bckpt_file_name = f'best_{checkpoint_name}'
        src_file = os.path.join(checkpoint_dir, checkpoint_name)
        dest_file = os.path.join(checkpoint_dir, bckpt_file_name)
        shutil.copy(src_file, dest_file)

        # Delete the old best checkpoint file
        if metadata.get('best_checkpoint') is not None:
            old_best_file = os.path.join(checkpoint_dir, metadata['best_checkpoint']['name'])
            if os.path.exists(old_best_file):
                os.remove(old_best_file)

        metadata['best_checkpoint'] = {
            'name': bckpt_file_name,
            'epoch': epoch,
            'val_loss': val_loss
        }
        logger.info(f"New best checkpoint updated: {bckpt_file_name}")


def manage_checkpoints(metadata, max_to_keep, ckpt_dir):
    """
    This function is to manage the checkpoints such as maximum number of checkpoint to be stored
    and updating the checkpoint.json meta file with latest checkpoint details. Delete the oldest checkpoint if the
    number of checkpoint that are stored reached the max_to_keep number.

    @param metadata: Meta information about all the checkpoint in the dict
    @param max_to_keep: Maximum number of the checkpoint need to be stored
    @param ckpt_dir: Path of the checkpoint directory
    """
    if len(metadata['checkpoints']) > max_to_keep:
        # Remove the oldest checkpoint
        old_checkpoint = metadata['checkpoints'].pop(0)
        old_ckpt_path = os.path.join(ckpt_dir, old_checkpoint['name'])
        if os.path.exists(old_ckpt_path):
            os.remove(old_ckpt_path)
            logger.info(f"Removed old checkpoint: {old_checkpoint['name']}")


def save_metadata(metadata, meta_file_path):
    """
    This function is store the updated metadata back to the checkpoint metadata file

    @param metadata: Updated metadata of the checkpoint
    @param meta_file_path: Path of the checkpoint metadata file
    """
    with open(meta_file_path, 'w') as meta_file:
        json.dump(metadata, meta_file, indent=4)
        logger.info(f"Metadata saved/updated in {meta_file_path}")


def save_checkpoint(model, optimizer, epoch, val_loss, max_to_keep=3, checkpoint_dir=None, prefix='',
                    best_model_metric='low'):
    """
    This function is to save and manage the checkpoint

    @param model: Model that need to be stored
    @param optimizer: Optimizer whose state need to be stored
    @param epoch: Current training epoch of the model
    @param val_loss: Current Validation loss of the model
    @param max_to_keep: Maximum number of the checkpoint need to stored at all time.
    @param checkpoint_dir: Path of the checkpoint directory
    @param prefix: Prefix for the checkpoint name
    @param best_model_metric: This param is used to indicate whether the high value is best or low value is best model.
    For example, during reward model the MSE will be used where the best model with least value
    whereas when ppo training the reward with maximum value is the best model
    """
    checkpoint_name = f"{prefix}checkpoint_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Save the model checkpoint
    save_model_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

    # Load existing metadata or initialize new
    meta_file_path = os.path.join(checkpoint_dir, 'checkpoint_meta.json')
    metadata = load_metadata(meta_file_path)

    is_best_model = False
    if metadata.get('best_checkpoint') is not None:
        if best_model_metric == 'low':
            is_best_model = val_loss < metadata['best_checkpoint']['val_loss']
        else:
            is_best_model = val_loss > metadata['best_checkpoint']['val_loss']
    # Update or add checkpoint metadata
    update_metadata(metadata, checkpoint_dir, checkpoint_name, epoch, val_loss, is_best_model)

    manage_checkpoints(metadata, max_to_keep, checkpoint_dir)

    # Save updated metadata
    save_metadata(metadata, meta_file_path)


def load_model(model, checkpoint_path):
    """
    This function is to load model from the checkpoint.

    @param model: Object of the pytorch model that need to be loaded
    @param checkpoint_path: Path of the checkpoint that need to be loaded
    @return: Return the model which loaded with checkpoint states
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    ckpt_epoch = checkpoint['epoch']
    ckpt_val_loss = checkpoint['val_loss']

    logger.info(f"Loaded model from epoch {ckpt_epoch} with validation loss {ckpt_val_loss}")
    return model


def load_checkpoint_by_path(model, optimizer, checkpoint_path):
    """
    This function is to load the model and optimizer from checkpoint using the provided path

    @param model: Object of the pytorch model that need to be loaded
    @param optimizer: Object of the optimizer that need to be loaded
    @param checkpoint_path: Path of the checkpoint that need to be loaded
    @return:
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ckpt_epoch = checkpoint['epoch']
    ckpt_val_loss = checkpoint['val_loss']

    logger.info(f"Loaded checkpoint from epoch {ckpt_epoch}, resuming from epoch {ckpt_epoch + 1}")
    return ckpt_epoch, ckpt_val_loss


def load_checkpoint(checkpoint_dir, model, optimizer, option='latest'):
    """
    This function is to load the latest checkpoint from the checkpoint directory using the checkpoint metadata file

    @param checkpoint_dir: Path of the checkpoint directory
    @param model: Object of the model that need to be loaded
    @param optimizer: Object of the optimizer that need to be loaded
    @param option: This parameter to provide whether the latest checkpoint or the best checkpoint need to be loaded
    @return: Return the restored checkpoint epoch and loss values
    """
    meta_file_path = os.path.join(checkpoint_dir, 'checkpoint_meta.json')

    if os.path.exists(meta_file_path):
        # Load metadata from the JSON file
        with open(meta_file_path, 'r') as meta_file:
            metadata = json.load(meta_file)

        if not metadata['checkpoints']:
            logger.info("No checkpoints listed in metadata, starting from scratch.")
            return 0, float('inf')

        # Load the latest checkpoint or best checkpoint based on the option
        if option == 'best':
            selected_checkpoint = metadata['best_checkpoint']
        else:
            selected_checkpoint = metadata['checkpoints'][-1]
        checkpoint_name = selected_checkpoint['name']
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        ckpt_epoch, ckpt_val_loss = load_checkpoint_by_path(model, optimizer, checkpoint_path)
        return ckpt_epoch, ckpt_val_loss
    else:
        logger.info("No checkpoint found, starting from scratch.")
        return 0, float('inf')
