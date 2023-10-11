import streamlit as st
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def plot_training_history(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    st.pyplot()


def display_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names)
    st.text(report)


st.title("Модель класифікації Fashion MNIST")


model_option = st.selectbox("Виберіть модель", ("MLP Model", "MLP Model Architecture 2", "MLP Model Architecture 3", "MLP Model Architecture 4"))


fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

validation_size = 4000
x_valid, x_train = x_train_full[:validation_size] / 255.0, x_train_full[validation_size:] / 255.0
y_valid, y_train = y_train_full[:validation_size], y_train_full[validation_size:]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


if st.button("Навчання моделі"):
    if model_option == "MLP Model":
        model = mlp_classifier()
    elif model_option == "MLP Model Architecture 2":
        model = mlp_classifier_architecture2()
    elif model_option == "MLP Model Architecture 3":
        model = mlp_classifier_architecture3()
    elif model_option == "MLP Model Architecture 4":
        model = mlp_classifier_architecture4()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

    st.write("Модель навчена!")


    _, accuracy = model.evaluate(x_test, y_test)
    st.write(f"Загальна точність: {accuracy:.2f}")


    st.subheader("Графік навчання")
    plot_training_history(history)


selected_index = st.number_input("Введіть індекс об'єкта для класифікації (0-9999):", min_value=0, max_value=9999, value=0)
if st.button("Виконати класифікацію"):
    if model_option == "MLP Model":
        model = mlp_classifier()
    elif model_option == "MLP Model Architecture 2":
        model = mlp_classifier_architecture2()
    elif model_option == "MLP Model Architecture 3":
        model = mlp_classifier_architecture3()
    elif model_option == "MLP Model Architecture 4":
        model = mlp_classifier_architecture4()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

    selected_image = x_test[selected_index]
    selected_label = y_test[selected_index]

    prediction = model.predict_classes(np.array([selected_image / 255.0]))
    st.image(selected_image, caption=f"Правильна мітка: {class_names[selected_label]}, "
                                     f"Передбачена мітка: {class_names[prediction[0]]}")


if st.button("Вивести звіт про класифікацію"):
    selected_image = x_test[selected_index]
    selected_label = y_test[selected_index]

    if model_option == "MLP Model":
        model = mlp_classifier()
    elif model_option == "MLP Model Architecture 2":
        model = mlp_classifier_architecture2()
    elif model_option == "MLP Model Architecture 3":
        model = mlp_classifier_architecture3()
    elif model_option == "MLP Model Architecture 4":
        model = mlp_classifier_architecture4()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

    prediction = model.predict_classes(np.array([selected_image / 255.0]))
    display_classification_report(np.array([selected_label]), prediction, class_names)
