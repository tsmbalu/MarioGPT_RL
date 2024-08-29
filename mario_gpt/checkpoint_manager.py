import logging
import os
import json
import shutil
import torch

logger = logging.getLogger(__name__)


def get_checkpoint_details(checkpoint_dir, checkpoint_name) -> (int, float):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path)
    checkpoint_epoch = checkpoint['epoch']
    checkpoint_val_loss = checkpoint['val_loss']
    return checkpoint_epoch, checkpoint_val_loss


def save_model_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")


def load_metadata(meta_file_path):
    if os.path.exists(meta_file_path):
        with open(meta_file_path, 'r') as meta_file:
            return json.load(meta_file)
    else:
        return {'checkpoints': [], 'best_checkpoint': None}


def update_metadata(metadata, checkpoint_dir, checkpoint_name, epoch, val_loss):
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
    if metadata.get('best_checkpoint') is None or val_loss < metadata['best_checkpoint']['val_loss']:
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
    if len(metadata['checkpoints']) > max_to_keep:
        # Remove the oldest checkpoint
        old_checkpoint = metadata['checkpoints'].pop(0)
        old_ckpt_path = os.path.join(ckpt_dir, old_checkpoint['name'])
        if os.path.exists(old_ckpt_path):
            os.remove(old_ckpt_path)
            logger.info(f"Removed old checkpoint: {old_checkpoint['name']}")


def save_metadata(metadata, meta_file_path):
    with open(meta_file_path, 'w') as meta_file:
        json.dump(metadata, meta_file, indent=4)
        logger.info(f"Metadata saved/updated in {meta_file_path}")


def save_checkpoint(model, optimizer, epoch, val_loss, max_to_keep=3, checkpoint_dir=None, prefix=''):
    checkpoint_name = f"{prefix}checkpoint_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Save the model checkpoint
    save_model_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

    # Load existing metadata or initialize new
    meta_file_path = os.path.join(checkpoint_dir, 'checkpoint_meta.json')
    metadata = load_metadata(meta_file_path)

    # Update or add checkpoint metadata
    update_metadata(metadata, checkpoint_dir, checkpoint_name, epoch, val_loss)

    manage_checkpoints(metadata, max_to_keep, checkpoint_dir)

    # Save updated metadata
    save_metadata(metadata, meta_file_path)


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    ckpt_epoch = checkpoint['epoch'] + 1
    ckpt_val_loss = checkpoint['val_loss']

    logger.info(f"Loaded model from epoch {ckpt_epoch} with validation loss {ckpt_val_loss}")
    return model


def load_checkpoint_by_path(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ckpt_epoch = checkpoint['epoch'] + 1
    ckpt_val_loss = checkpoint['val_loss']

    logger.info(f"Loaded checkpoint from epoch {ckpt_epoch}, resuming from epoch {ckpt_epoch + 1}")
    return ckpt_epoch, ckpt_val_loss


def load_checkpoint(checkpoint_dir, model, optimizer, option='latest'):
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
