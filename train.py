from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

def main():
    model = Sequential()
    # 畳み込み層
    # 画像処理を3*3で行う 入力は64*64でカラー画像3(赤・青・緑)
    model.add(Conv2D(64,(3,3), input_shape=(64,64,3)))
    model.add(Activation("relu"))
    # MaxPooling画像の特定の領域から最大値を抽出する操作
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2段目の畳み込み層
    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ニューロン作成
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    # 出力を作成
    model.add(Dense(3))
    model.add(Activation("softmax"))
    model.summary()

    # フィッティングの（重み）調整の実施
    model.compile(
            optimizer="adam",
            # 分類に関して
            loss="categorical_crossentropy",
            metrics=["accuracy"])
    # 画像をニューラルネットワーク内の数字0~1になおす
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    # 学習用のデータ
    train_generator = train_datagen.flow_from_directory(
          "data/train",
          # 読み込んだ画像をリサイズ
          target_size=(64,64),
          # 一括に処理する数を指定
          batch_size=8)
          # テスト用のデータ
    validation_generator = test_datagen.flow_from_directory(
      "data/validation",
          target_size=(64,64),
          batch_size=10)
    model.fit(
            train_generator,
            # 何回繰り返しフィッティングを行うか（やりすぎると過学習になる）
            epochs=30,
            # １epochs内に行うフィッティングの回数を指定
            steps_per_epoch=10,
            validation_data=validation_generator,
            # 何回に1回テストするかを指定する
            validation_steps=10)
    model.save("model.h5")
    # val_accの利率を確認する


if __name__ == "__main__":
    main()