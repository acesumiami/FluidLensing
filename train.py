import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    LearningRateScheduler
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.metrics import CategoricalAccuracy, MeanIoU
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train an Attentive ResUNet model for coral reef segmentation')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--train_lbl_dir', type=str, required=True, help='Directory containing training labels')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Directory containing validation images')
    parser.add_argument('--val_lbl_dir', type=str, required=True, help='Directory containing validation labels')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
    parser.add_argument('--image_size', type=int, default=512, help='Image dimension for training (square)')
    parser.add_argument('--channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--initial_lr', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--max_weight', type=float, default=10.0, help='Maximum class weight')
    parser.add_argument('--reef_boost', type=float, default=1.5, help='Boosting factor for reef classes')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='L2 regularization factor')
    
    return parser.parse_args()


def define_classes():
    classes = {
        0: {"name": "Branching Coral", "hex": "#93e7a9"},
        1: {"name": "Massive Coral",   "hex": "#4366d4"},
        2: {"name": "Coral Fore Reef", "hex": "#08e8f6"},
        3: {"name": "Reef Crest - Coralline Algal Ridge", "hex": "#f2bc88"},
        4: {"name": "Algae",           "hex": "#5dbd3f"},
        5: {"name": "Sand / Rubble",   "hex": "#b0447c"},
        6: {"name": "No Class",        "hex": "#9c7261"}
    }
    
    num_classes = len(classes)
    class_names = [classes[i]["name"] for i in range(num_classes)]
    class_hex_colors = [classes[i]["hex"] for i in range(num_classes)]
    
    color_map = {
        tuple(int(info["hex"].lstrip("#")[i:i+2], 16) for i in (0,2,4)): cid
        for cid, info in classes.items()
    }
    
    return classes, num_classes, class_names, class_hex_colors, color_map


def rgb_to_onehot(rgb, color_map, num_classes, image_height, image_width):
    flat = tf.reshape(tf.cast(rgb, tf.int32), [-1,3])
    cmap = tf.constant(list(color_map.keys()), dtype=tf.int32)
    matches = tf.reduce_all(flat[:,None,:] == cmap[None,:,:], axis=2)
    labels = tf.argmax(tf.cast(matches, tf.int32), axis=1)
    labels = tf.reshape(labels, [image_height, image_width])
    return tf.one_hot(labels, num_classes)


def load_pair(img_path, lbl_path, color_map, num_classes, image_height, image_width):
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
    img = tf.image.resize(img, [image_height, image_width]) / 255.0
    lbl = tf.image.decode_png(tf.io.read_file(lbl_path), channels=3)
    lbl = tf.image.resize(lbl, [image_height, image_width], method="nearest")
    lbl = rgb_to_onehot(lbl, color_map, num_classes, image_height, image_width)
    return img, lbl


def valid_filter(img, lbl):
    white = tf.reduce_all(img > 0.98, axis=-1)
    return tf.reduce_mean(tf.cast(white, tf.float32)) < 0.25


def get_file_list(folder):
    return sorted(str(p) for p in Path(folder).iterdir()
                  if p.suffix.lower() in (".png",".jpg",".jpeg"))


def make_ds(img_dir, lbl_dir, color_map, num_classes, image_height, image_width, batch_size, augment=False):
    imgs = get_file_list(img_dir)
    lbls = get_file_list(lbl_dir)
    ds = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    if augment:
        ds = ds.shuffle(1000)
        
    ds = ds.map(lambda i,l: load_pair(i, l, color_map, num_classes, image_height, image_width), 
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(valid_filter)
    ds = ds.map(lambda i,l: (tf.image.per_image_standardization(i), l),
                num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug_args = dict(
            rotation_range=10, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.1,
            zoom_range=0.1, horizontal_flip=True,
            fill_mode='nearest'
        )
        img_gen = ImageDataGenerator(**aug_args)
        lbl_gen = ImageDataGenerator(**aug_args)

        def aug_fn(i, l):
            seed = np.random.randint(1e6)
            return (img_gen.random_transform(i.numpy(), seed=seed),
                    lbl_gen.random_transform(l.numpy(), seed=seed))

        ds = ds.map(lambda i,l: tf.py_function(
                        func=aug_fn, inp=[i,l],
                        Tout=[tf.float32, tf.float32]
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE)

        def set_shapes(i, l):
            i.set_shape([image_height, image_width, 3])
            l.set_shape([image_height, image_width, num_classes])
            return i, l

        ds = ds.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_weights(ds, num_classes, max_weight):
    total = 0
    counts = np.zeros(num_classes, np.float64)
    for _, masks in ds.unbatch().batch(1):
        m = masks.numpy()
        counts += m.sum(axis=(0,1,2))
        total += np.prod(m.shape[1:3])
    freq = counts / total
    med = np.median(freq[np.isfinite(freq)])
    weights = med / freq
    weights = np.where(np.isfinite(weights), weights, med)
    return np.clip(weights, None, max_weight)


def masked_weighted_cce(class_weights):
    cw = tf.constant(class_weights, dtype=tf.float32)
    cce_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction="none"
    )
    def loss(y_true, y_pred):
        per_pixel = cce_fn(y_true, y_pred)
        mask = tf.reduce_any(y_true > 0.5, axis=-1)
        mask = tf.cast(mask, tf.float32)
        labels = tf.argmax(y_true, axis=-1)
        weights = tf.gather(cw, labels)
        weighted = per_pixel * weights * mask
        return tf.reduce_sum(weighted) / (tf.reduce_sum(mask) + 1e-6)
    return loss


def ResidualBlock(x, filters, l2_reg):
    reg = regularizers.l2(l2_reg)
    shortcut = x
    in_ch = x.shape[-1]
    if in_ch != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same",
                                 kernel_regularizer=reg)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def AttentionGate(x, g, inter_channels, l2_reg):
    reg = regularizers.l2(l2_reg)
    theta_x = layers.Conv2D(inter_channels, 1, padding="same",
                            kernel_regularizer=reg)(x)
    phi_g = layers.Conv2D(inter_channels, 1, padding="same",
                            kernel_regularizer=reg)(g)
    f = layers.Activation("relu")(layers.add([theta_x, phi_g]))
    psi = layers.Conv2D(1, 1, padding="same",
                            kernel_regularizer=reg)(f)
    psi = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([x, psi])


def build_attentive_resunet(input_shape, num_classes, l2_reg):
    backbone = EfficientNetB3(include_top=False,
                              input_shape=input_shape,
                              weights="imagenet")
    inputs = backbone.input
    skip_names = [
        "block2a_expand_activation",
        "block3a_expand_activation",
        "block4a_expand_activation",
        "block6a_expand_activation"
    ]
    skips = [backbone.get_layer(n).output for n in skip_names]
    x = backbone.output

    decoder_filters = [512, 256, 128, 64]
    for f, skip in zip(decoder_filters, reversed(skips)):
        x = layers.UpSampling2D(2)(x)
        attn = AttentionGate(skip, x, inter_channels=f//2, l2_reg=l2_reg)
        x = layers.Concatenate()([x, attn])
        x = ResidualBlock(x, f, l2_reg=l2_reg)
        x = ResidualBlock(x, f, l2_reg=l2_reg)

    x = layers.UpSampling2D(2)(x)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax",
                            kernel_regularizer=regularizers.l2(l2_reg))(x)
    return Model(inputs, outputs)


def lr_scheduler(epoch, lr):
    if epoch in (10, 15):
        return lr * 0.5
    return lr


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    classes, num_classes, class_names, class_hex_colors, color_map = define_classes()
    
    train_ds = make_ds(args.train_img_dir, args.train_lbl_dir, color_map, num_classes,
                       args.image_size, args.image_size, args.batch_size, augment=True)
    val_ds = make_ds(args.val_img_dir, args.val_lbl_dir, color_map, num_classes,
                     args.image_size, args.image_size, args.batch_size, augment=False)
    
    orig_w = compute_weights(train_ds, num_classes, args.max_weight)
    focus_ids = [0, 1, 2, 3]
    adj_w = orig_w.copy()
    adj_w[focus_ids] = np.clip(adj_w[focus_ids] * args.reef_boost, None, args.max_weight)
    
    model = build_attentive_resunet(
        input_shape=(args.image_size, args.image_size, args.channels),
        num_classes=num_classes,
        l2_reg=args.l2_reg
    )
    
    model.compile(
        optimizer=Adam(args.initial_lr),
        loss=masked_weighted_cce(adj_w),
        metrics=[CategoricalAccuracy(name="cat_acc"),
                 MeanIoU(num_classes=num_classes, name="mean_io_u")]
    )
    
    lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)
    
    callbacks = [
        ModelCheckpoint(os.path.join(args.output_dir, "best_underwater_seg.keras"),
                        monitor="val_cat_acc",
                        save_best_only=True,
                        mode="max", verbose=1),
        EarlyStopping(monitor="val_cat_acc",
                      patience=10,
                      restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="loss",
                          factor=0.5,
                          patience=5,
                          min_lr=1e-6,
                          verbose=1),
        lr_callback
    ]
    
    print(f"Training model with the following configuration:")
    print(f"- Image size: {args.image_size}x{args.image_size}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Initial learning rate: {args.initial_lr}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Class weights: {adj_w}")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['cat_acc'], label='Training Accuracy')
    plt.plot(history.history['val_cat_acc'], label='Validation Accuracy')
    plt.title('Categorical Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=300)
    
    loss, cat_acc, miou = model.evaluate(val_ds)
    print(f"Val Loss: {loss:.4f}   CatAcc: {cat_acc:.4f}   MeanIoU: {miou:.4f}")
    
    results = {
        'final_val_loss': float(loss),
        'final_val_accuracy': float(cat_acc), 
        'final_val_miou': float(miou),
        'num_classes': num_classes,
        'class_names': class_names,
        'class_hex_colors': class_hex_colors,
        'image_size': args.image_size
    }
    
    import json
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Training complete. Model and results saved to {args.output_dir}")


if __name__ == "__main__":
    main()