'''
Training script based on the RetinaNet Keras tutorial, customized for AML cell detection task using Pascal VOC format.
'''

import os
import argparse
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.callbacks import RedirectModel, Evaluate
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaNet on AML cell images using Pascal VOC format.')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory containing JPEGImages, Annotations, ImageSets/Main')
    parser.add_argument('--snapshot-path', type=str, default='./snapshots',
                        help='Directory to store model checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size per training step')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone model to use (resnet50, resnet101, etc.)')
    return parser.parse_args()

def create_generators(data_dir, batch_size):
    train_generator = PascalVocGenerator(
        data_dir,
        'train',
        skip_difficult=True,
        shuffle_groups=True,
        batch_size=batch_size,
        image_min_side=800,
        image_max_side=800,
        preprocess_image=preprocess_image
    )

    val_generator = PascalVocGenerator(
        data_dir,
        'val',
        skip_difficult=True,
        shuffle_groups=False,
        batch_size=1,
        image_min_side=800,
        image_max_side=800,
        preprocess_image=preprocess_image
    )

    return train_generator, val_generator

def main():
    args = parse_args()

    # Create snapshot dir
    os.makedirs(args.snapshot_path, exist_ok=True)

    # Create generators
    train_gen, val_gen = create_generators(args.data_dir, args.batch_size)

    # Load base model (pre-trained on COCO)
    model = models.backbone(args.backbone).retinanet(num_classes=train_gen.num_classes())
    model.compile(
        loss={
            'regression': models.losses.smooth_l1(),
            'classification': models.losses.focal()
        },
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(args.snapshot_path, 'resnet50_{epoch:02d}.h5'),
            save_best_only=True,
            monitor='loss',
            mode='min',
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(args.snapshot_path, 'logs'),
            histogram_freq=0,
            write_graph=True,
            write_images=False
        ),
        RedirectModel(
            Evaluate(val_gen, tensorboard=os.path.join(args.snapshot_path, 'logs')),
            model
        )
    ]

    print(f"[INFO] Starting training for {args.epochs} epochs...")
    model.fit(
        train_gen,
        epochs=args.epochs,
        steps_per_epoch=1000,  # or len(train_gen) for full epoch
        callbacks=callbacks,
        verbose=1
    )

    print("[INFO] Training complete. Model checkpoints saved in:", args.snapshot_path)

if __name__ == '__main__':
    main()
