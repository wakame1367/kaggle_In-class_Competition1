from base.base_model import BaseModel
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.nasnet import NASNetLarge
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Dropout, GlobalAveragePooling2D
from keras import backend as k


class ConvMnistModel(BaseModel):
    def __init__(self, config):
        super(ConvMnistModel, self).__init__(config)
        self.input_shape = (56, 56)
        self.num_classes = 10
        self.build_model()

    def build_model(self):
        input_image = Input(shape=self.input_shape)
        input_image_ = Lambda(lambda x:
                              k.repeat_elements(k.expand_dims(x, 3), 3, 3))(input_image)
        base_model = MobileNetV2(input_tensor=input_image_,
                                 include_top=False,
                                 pooling='avg')
        output = Dropout(0.5)(base_model.output)
        predict = Dense(self.num_classes,
                        activation='softmax')(output)

        self.model = Model(inputs=input_image,
                           outputs=predict)
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])


class NasNetModel(BaseModel):
    def __init__(self, config):
        super(NasNetModel, self).__init__(config)
        self.input_shape = (32, 32)
        self.num_classes = 10
        self.build_model()

    def build_model(self):
        input_image = Input(shape=self.input_shape)
        input_image_ = Lambda(lambda x:
                              k.repeat_elements(k.expand_dims(x, 3), 3, 3))(input_image)
        base_model = NASNetLarge(input_tensor=input_image_,
                                 include_top=False,
                                 pooling='avg')
        output = Dropout(0.5)(base_model.output)
        predict = Dense(self.num_classes,
                        activation='softmax')(output)

        self.model = Model(inputs=input_image,
                           outputs=predict)
        self.model.summary()
        self.model.compile(optimizer='sgd',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
