# code inspired by
# https://pyimagesearch.com/2020/10/19/adversarial-images-and-attacks-with-keras-and-tensorflow/#:~:text=Adversarial%20images%20are%20perturbed%20in,identical%20to%20the%20human%20eye.

# import necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers.legacy import Adam
import keras.utils as kimage

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2

# hyperparameters and constants
LR = 5
max_images = 5000
steps = 1000
batch_size = 30
image_folder = "./imagenet"
output_folder = "./outputs"
img_size = 224
class_idx = 294 # Brown Bear
# class_idx = 430 # Basketball

"""
Optimizes the pixels in the patch to produce an adversarial sticker
"""
def generate_sticker(model, lossfn, optimizer, baseImages, sticker, classIdx, steps, batch_size):
    for step in range(0, steps):
        with tf.GradientTape() as tape:
            # track sticker for gradient descent
            tape.watch(sticker)

            _, overlay = overlay_sticker(baseImages, sticker, count=batch_size)
            adversary = preprocess_input(overlay)
            
            # predict the class of image with sticker and compute the loss wrt desired class
            predictions = model(adversary, training=False)
            loss = sum(lossfn(tf.convert_to_tensor([classIdx]), predictions[i]) for i in range(batch_size))
            
            print(f"step: {step}, loss: {loss.numpy() / batch_size} ...")
        # use loss to run the optimizer
        gradients = tape.gradient(loss, sticker)
        optimizer.apply_gradients([(gradients, sticker)])

    return sticker

"""
Overlays stickers on top of provided images
"""
def overlay_sticker(all_images, sticker, dims=None, x=None, y=None, i=None, scale=None, theta=None, noise=5, count=1):
    if dims is None:
        dims = tf.repeat([[float(all_images.shape[1]), float(all_images.shape[2])]], all_images.shape[0], axis=0)
    
    if i is None:
        i = np.random.randint(0, all_images.shape[0], size=count)
    else:
        i = np.array([i])

    images = tf.gather(all_images, i)
    dims = tf.gather(dims, i)

    y_factor = dims[:, 0] / images.shape[1]
    x_factor = dims[:, 1] / images.shape[2]

    if scale is None:
        scale = tf.constant(np.random.uniform(size=(count,), low=0.4, high=1.2), dtype=tf.float32)
    else:
        scale = tf.ones(count) * 1/scale

    rpad = images.shape[1] - sticker.shape[1]
    cpad = images.shape[2] - sticker.shape[2]

    sx = scale * x_factor
    sy = scale * y_factor

    if x is None:
        x = tf.constant(np.random.uniform(size=(count,),
                              low=-images.shape[2]/2 + sticker.shape[2]/2/sx, 
                              high= images.shape[2]/2 - sticker.shape[2]/2/sx), dtype=tf.float32,)
    else:
        x = tf.ones(count) * x

    if y is None:
        y = tf.constant(np.random.uniform(size=(count,),
                              low=-images.shape[1]/2 + sticker.shape[1]/2/sy,
                              high= images.shape[1]/2 - sticker.shape[1]/2/sy), dtype=tf.float32,)
    else:
        y = tf.ones(count) * y


    if theta is None:
        theta = tf.constant(np.random.normal(size=(count,), loc=0, scale=0.2), dtype=tf.float32,)
    else:
        theta = tf.ones(count) * theta

    sticker = tf.repeat(sticker, count, axis=0)
    
    matte = tf.ones_like(sticker)
    sticker = tf.pad(sticker, [[0, 0], [int(np.floor(rpad/2)), int(np.ceil(rpad/2))], [int(np.floor(cpad/2)), int(np.ceil(cpad/2))], [0, 0]])
    matte = tf.pad(matte, [[0, 0], [int(np.floor(rpad/2)), int(np.ceil(rpad/2))], [int(np.floor(cpad/2)), int(np.ceil(cpad/2))], [0, 0]])
    
    
    zero = tf.zeros(count)
    one = tf.ones(count)
    shift_x = tf.ones(count) * images.shape[2]/2
    shift_y = tf.ones(count) * images.shape[1]/2
    cos = tf.cos(theta)
    sin = tf.sin(theta)
    
    # apply scale and rotations
    transform = tf.stack([sx * cos, -sy * sin, -shift_x * (sx * cos - sy * sin - one),
                          sx * sin, sy * cos, -shift_y * (sx * sin + sy * cos - one),
                          zero, zero], axis=1)
    sticker = tfa.image.transform(sticker, transform)
    matte = tfa.image.transform(matte, transform)
    # apply translations
    translate = tf.stack([x, y], axis=1)
    sticker = tfa.image.translate(sticker, translate)
    matte = tfa.image.translate(matte, translate)

    # change range of sticker to be between darkest abd brightest intensity values
    intensities = 1/3 * tf.math.reduce_sum(images, axis=3)
    bright = tf.math.reduce_max(intensities, axis=[1,2]) * .8
    dark = tf.math.reduce_min(intensities, axis=[1,2]) * 1.2
    rnge = tf.reshape(tf.repeat(bright - dark, 224 * 224 * 3, axis=0), (images.shape[0], 224, 224, 3))
    sticker = sticker / 255 * rnge

    applied = sticker * matte + images * (1 - matte)

    # add random noise
    noise_amt = np.random.randint(0, noise)
    applied += np.random.normal(size=applied.shape, loc=0, scale=noise_amt)
    return i, tf.clip_by_value(applied, 0, 255)


def main():
    print("loading images...")
    all_images = np.zeros((max_images, img_size, img_size, 3))
    # keep track of original image dimensions
    dims = np.zeros((max_images, 2))
    for i in range(max_images):
        path = f"{image_folder}/image_{str(i + 1).zfill(5)}.JPEG"
        image = kimage.load_img(path)
        image = kimage.img_to_array(image)
        dims[i] = [float(image.shape[0]), float(image.shape[1])]
        image = cv2.resize(image, (img_size, img_size))
        all_images[i] = image
        

    # load the pre-trained ResNet50 model for running inference
    print("loading pre-trained ResNet50 model...")
    model = ResNet50(weights="imagenet")
    optimizer = Adam(learning_rate=LR)
    sccLoss = SparseCategoricalCrossentropy()

    # create a tensor for all the images
    baseImages = tf.constant(all_images, dtype=tf.float32)
    patch = tf.random.uniform((1, 50, 50, 3)) * 255 # initialize as random RGB values
    sticker = tf.Variable(patch, trainable=True)

    # generate the perturbation vector to create an adversarial example
    print("generating sticker...")
    finalSticker = generate_sticker(model, sccLoss, optimizer, baseImages, sticker, class_idx, steps, batch_size)
    stickerImage = sticker.numpy().squeeze()
    stickerImage = np.clip(stickerImage, 0, 255).astype("uint8")
    stickerImage = cv2.cvtColor(stickerImage, cv2.COLOR_RGB2BGR)

    # save the sticker
    cv2.imwrite(f"{output_folder}/sticker.png", stickerImage)

    # run a test example of the sticker on an image and display result
    imageId, withSticker = overlay_sticker(baseImages, finalSticker)
    print("predicting label for sticker on image " + str(imageId))

    preprocessedImage = preprocess_input(withSticker)
    predictions = model.predict(preprocessedImage)
    predictions = decode_predictions(predictions, top=3)[0]

    label = predictions[0][1]
    confidence = round(predictions[0][2] * 100, 2)

    print(f"classified as label: {label} with confidence: {confidence}")

    # save the image with the sticker and its predicted label/confidence
    adverImage = withSticker.numpy().squeeze()
    adverImage = np.clip(adverImage, 0, 255).astype("uint8")
    adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_folder}/result.png", adverImage)
    cv2.putText(adverImage, f"{label}: {confidence}", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(f"{output_folder}/result_classified.png", adverImage)

if __name__ == "__main__":
    main()