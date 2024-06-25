import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load your pre-trained model
loaded_model_imageNet = load_model("scene_detection.h5")
disease_dic= ['Forest', 'Glacier', 'Mountain','Sea','Street']
def pred_leaf_disease(frame):				 
    resized_frame = cv2.resize(frame, (256, 256))
    x = np.expand_dims(resized_frame, axis=0)
    x = preprocess_input(x)
    result = loaded_model_imageNet.predict(x)
    print((result*100).astype('int'))
    final_list_result=(result*100).astype('int')
    list_vals=list(final_list_result[0])
    result_val=max(list(final_list_result[0]))
    print(result_val)
    index_result = list_vals.index(result_val)
    return index_result

def capture_and_predict():
    cap = cv2.VideoCapture(0)  # 0 is the default index of the webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        index_result = pred_leaf_disease(frame)
        prediction = (str(disease_dic[index_result]))

        print("The background scene is ",prediction)

        cv2.putText(frame, f"Prediction: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to start capturing and predicting in real-time
#capture_and_predict()
