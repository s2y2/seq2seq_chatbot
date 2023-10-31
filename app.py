
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import numpy as np
import re
import pickle
from rouge import Rouge
from flask import Flask, render_template, request

#Load pairs from the file
with open('pairs.pkl', 'rb') as f:
    pairs= pickle.load(f) 

latent_dim = 256

#Initialize empty lists to hold sentences
input_docs = []
target_docs = []
#Initialize empty sets for vocabulary
input_tokens = set()
target_tokens = set()

for line in pairs[:1000]:
  input_doc, target_doc = line[0], line[1]
  #Append each input sentence to input_docs
  input_docs.append(input_doc)
  #Split words from punctuation in target_doc
  target_doc = " ".join([word for word in re.findall(r"[\w']+", target_doc)])
  # Redefine target_doc and append it to target_docs
  target_doc = '<START> ' + target_doc + ' <END>'
  target_docs.append(target_doc)

  # Split each sentence into words and add unique words to vocabulary sets
  for token in re.findall(r"[\w']+", input_doc):
    # Add your code here:
    if any(char.isalnum() for char in token):
      if token not in input_tokens:
        input_tokens.add(token)
  for token in target_doc.split():
    # And here:
    if any(char.isalnum() for char in token):
      if token not in target_tokens:
        target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

#Create num_encoder_tokens and num_decoder_tokens
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

#Find maximum sequence lengths
max_encoder_seq_length = max([len(re.findall(r"[\w']+", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+", target_doc)) for target_doc in target_docs])

#Create dictionaries to map tokens to indices
input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

#Initialize encoder input data
encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

#Populate encoder input data
for line, input_doc in enumerate(input_docs):
  for timestep, token in enumerate(re.findall(r"[\w']+", input_doc)):
    encoder_input_data[line, timestep, input_features_dict[token]] = 1.

#Restore the pre-trained model and construct the encoder and decoder models
model_uploaded = load_model('training_model.h5')

print("model_uploaded")
model_uploaded.summary()

encoder_inputs = model_uploaded.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model_uploaded.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model_uploaded.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model_uploaded.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model_uploaded.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

#Create dictionaries for reverse mapping
reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

class ChatBot():

  def __init__(self):
        self.conversation_history = []  #Initialize an empty conversation history

  def decode_sequence(self, input_seq):
      #Encode the input as state vectors.
      states_value = encoder_model.predict(input_seq)

      #Generate empty target sequence of length 1
      target_seq = np.zeros((1, 1, num_decoder_tokens))
      #Populate the first token of target sequence with the start token
      target_seq[0, 0, target_features_dict['<START>']] = 1.

      #Sampling loop for a batch of sequences
      prev_word = None  #Variable to keep track of the previous word
      decoded_words = []

      stop_condition = False
      while not stop_condition:
        #Run the decoder model to get possible output tokens and states
        output_tokens, hidden_state, cell_state = decoder_model.predict(
          [target_seq] + states_value)

        #Choose token with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        
        #Exit condition: either hit max length or find stop token.
        if (sampled_token == '<END>' or len(decoded_words) >= max_decoder_seq_length):
          stop_condition = True

        if sampled_token != prev_word:
            decoded_words.append(sampled_token)
            prev_word = sampled_token
        #Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        #Update states
        states_value = [hidden_state, cell_state]


      #Remove the trailing "<END>" token if it exists
      if decoded_words and decoded_words[-1] == '<END>':
        decoded_words.pop()

      return " ".join(decoded_words)

  def generate_response(self, user_input):
        #Append the user's input to the conversation history
        self.conversation_history.append(user_input)

        #Concatenate the entire conversation history
        conversation_text = " ".join(self.conversation_history)

        #Process the concatenated conversation text and update input_seq
        input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        for timestep, token in enumerate(re.findall(r"[\w']+", conversation_text)):
            if any(char.isalnum() for char in token):
                if token in input_features_dict:
                    timestep = min(timestep, max_encoder_seq_length - 1)
                    input_seq[0, timestep, input_features_dict[token]] = 1.0

        decoded_response = self.decode_sequence(input_seq)

        return decoded_response

  def generate_and_evaluate_response(self, dataset):
        model_out_list = []
        reference_list = []

        for input_query, reference_response in dataset:
          self.conversation_history = []

          #Generate the response
          generated_response = self.generate_response(input_query)
          model_out_list.append(generated_response)
          reference_list.append(reference_response)
          #Evaluate the generated response using ROUGE

        print("model_out_list")
        print(model_out_list)
        print("reference_list")
        print(reference_list)

        rouge = Rouge()
        rouge_output = rouge.get_scores(model_out_list, reference_list, avg=True)
        return rouge_output 

#Instantiate the ChatBot object   
chatbot_object = ChatBot()

#Define a dataset for evaluation
dataset = [
    ("hi, how are you doing?", "I'm fine. How about yourself?"),
    ("How's it going?", "I'm doing well. How about you?"),
    ("I'm doing well. How about you?","Never better, thanks"),
    ("So how have you been lately?","I've actually been pretty good. You?"),
    ("it's such a nice day","Yes, it is"),
    ("where are you going to school?","I'm going to")
]

#Evaluate and print the chatbot's response
print("Evaluation:")
print(chatbot_object.generate_and_evaluate_response(dataset))

#Initialize a Flask web application
app = Flask(__name__)

#Define a route for the home page
@app.route("/")
def home():
    return render_template("index.html")

#Define a route for handling user input and getting the bot's response
@app.route("/get")
def get_bot_response(): 
    chatbot_object = ChatBot()
    user_input = request.args.get('msg')
    return str(chatbot_object.generate_response(user_input))

#Start the Flask application if this script is executed
if __name__ == "__main__":
    app.run(debug=True) 