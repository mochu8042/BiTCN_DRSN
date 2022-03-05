import os
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Dropout, Flatten, Concatenate, Activation
from application import parse_file, get_vectors_df
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, GlobalMaxPooling1D
from tcn import TCN
import time
import datetime
from my_parse import My_Parse

parse = My_Parse()
args = parse.get_args()
print(args.size)


def reverse_tensor(inputs, axis):
    return K.reverse(inputs, axis)


def read_data():
    filename = args.filename
    parse_file(filename)
    base = os.path.splitext(os.path.basename(filename))[0]

    vector_filename = "pkl/" + base + "_gadget_vectors.pkl"
    vector_length = args.len
    if os.path.exists(vector_filename):
        df = joblib.load(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
        joblib.dump(df, vector_filename)

    vectors = np.stack(df.iloc[:, 1].values)
    labels = df.iloc[:, 0].values
    positive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=False)
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

    X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs], labels[resampled_idxs], train_size=0.8,
                                                        test_size=0.2, stratify=labels[resampled_idxs])
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test, labels


class TextCNN(object):

    def get_model(self):
        inputs = Input((train_x.shape[1], train_x.shape[2]))
        x1 = TCN(args.num, dilations=(1, 2, 4, 8, 16), return_sequences=True, kernel_size=args.size)(inputs)
        x1 = GlobalMaxPooling1D()(x1)
        x2 = Lambda(lambda x: K.reverse(x, axes=1))(inputs)
        x2 = TCN(args.num, dilations=(1, 2, 4, 8, 16), return_sequences=True, kernel_size=args.size)(x2)
        x2 = GlobalMaxPooling1D()(x2)
        x1 = Dense(x1.get_shape().as_list()[-1], activation="relu")(x1)
        x2 = Dense(x2.get_shape().as_list()[-1], activation="relu")(x2)
        x = Concatenate()([x1, x2])
        x = Dropout(args.dropout)(x)
        output = Dense(2, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=output)
        return model


train_x, train_y, test_x, test_y, labels = read_data()
print('Build model...')
model = TextCNN().get_model()
model.compile('nadam', 'binary_crossentropy', metrics=['accuracy'])

print("数据集:" + args.filename)
print('Train...')
class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)

start = time.clock()
H = model.fit(train_x, train_y,
              batch_size=args.batch_size,
              epochs=args.epoch,
              class_weight=class_weight)
end = time.clock()
train_time = datetime.timedelta(seconds=end - start)

start = time.clock()
values = model.evaluate(test_x, test_y, args.batch_size)
end = time.clock()
test_time = datetime.timedelta(seconds=end - start)
print("Accuracy is...", values[1])
predictions = (model.predict(test_x, batch_size=args.batch_size)).round()

tn, fp, fn, tp = confusion_matrix(np.argmax(test_y, axis=1), np.argmax(predictions, axis=1)).ravel()
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
print('False positive rate is...', fpr)
print('False negative rate is...', fnr)
recall = tp / (tp + fn)
print('True positive rate is...', recall)
precision = tp / (tp + fp)
print('Precision is...', precision)
f1 = (2 * precision * recall) / (precision + recall)
print('F1 score is...', f1)
print("训练时间：", train_time)
print("测试时间：", test_time)
loss = H.history["loss"]
val_loss = H.history["val_loss"]
acc = H.history["accuracy"]
val_acc = H.history["val_accuracy"]
