import numpy as np
import tensorflow as tf
def interpre(data):
    interpreter = tf.lite.Interpreter(
        model_path='model/handtrain(400(91)).tflite',
        num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result=np.squeeze(output_data)
    return result