import numpy as np
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import inception_v3
from keras.optimizers import SGD
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from settings import *


# TODO: la size non dovrebbe essere hardcoded
def preprocess(img, params):
    img = cv2.resize(img, (299, 299))
    if img.shape == (299, 299):
        img = np.repeat(img, 3).reshape(299, 299, 3)
    img = ((img / 255) - 0.5) * 2
    return img


# TODO: non mi cago callbacks
# TODO: aggiungere parametri per training
def train(model, train_data, valid_data):
    # training 1
    for layer in model.layers[:311]:  # inceptionv3 ha 311 livelli
        layer.trainable = False
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    callbacks = [EarlyStopping('val_loss', patience=2)]
    callbacks += [
        ModelCheckpoint("out\\weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True,
                        save_weights_only=True, mode='auto')]
    history1 = model.fit_generator(
        train_data, validation_data=valid_data,
        callbacks=callbacks,
        epochs=3)

    plot_x = list(range(1, len(history1.history['val_acc']) + 1))
    plot_y = history1.history['val_acc']

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(0.0, 3)
    plt.ylim(0.0, 1.0)
    print(plot_x)
    print(plot_y)
    plt.plot(plot_x, plot_y, color='blue', linestyle='-')
    plt.savefig('out\\plot1.png', dpi=300)

    # training 2
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history2 = model.fit_generator(
        train_data, validation_data=train_data,
        callbacks=[EarlyStopping('val_loss', patience=2)],
        epochs=1)

    plot_x = list(range(1, len(history2.history['val_acc']) + 1))
    plot_y = history2.history['val_acc']

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(0.0, 3)
    plt.ylim(0.0, 1.0)
    plt.plot(plot_x, plot_y, color='blue', linestyle='-')
    plt.savefig('out\\plot2.png', dpi=300)


def create(imagenet_weights=True):
    weights = 'imagenet' if imagenet_weights else None
    inception = inception_v3.InceptionV3(include_top=False, weights=weights,
                                         input_shape=(299, 299, 3))
    x = inception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CATEGORIES, activation='softmax')(x)
    model = Model(inputs=inception.input, outputs=predictions)
    return model
