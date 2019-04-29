import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

##########################  YOUR CODE BEGINS HERE ##############################
# TODO [YOU] - add any imports required.

##########################  YOUR CODE ENDS HERE ################################


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()

        #################### YOUR CODE BEGINS HERE ##################################
        # TODO [YOU] - create the encoder for this model.
        resnet = None
        modules = list(resnet.children())[:-1]  # This deletes the last fully connected layer of resnet 152
        self.resnet = nn.Sequential(*modules)
        self.linear = None
        self.bn = None
        #################### YOUR CODE ENDS HERE ####################################

    def forward(self, images):
        """Extract feature vectors from input images."""

        #################### YOUR CODE BEGINS HERE ##################################
        # TODO [YOU] - perform the forward pass on this model.
        with torch.no_grad():
            features = None

        features = None  # Reshape the features so that the image features are flattened to give vectors
        features = None  # Apply the Linear Layer
        features = None  # Apply Batch Normalization
        #################### YOUR CODE ENDS HERE ####################################

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        #################### YOUR CODE BEGINS HERE ##################################
        # TODO [YOU] - perform the forward pass on this model.
        embeddings = None    # Get the embeddings corresponding to the words of the caption
        embeddings = None    # You can think of the image features as the 'first word' of the captionn itself. Combine the image features with the caption embeddings before passing to LSTM
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # This allows the LSTM to handle variable length sequences by considering the padding on the excess length
        packed_hiddens, _ = (None, None)  # apply the lstm. keep the activations of the final hidden state
        hiddens = packed_hiddens[0]
        outputs = None  # Apply the linear layer to get output of with Vocabulary size as last dimension
        #################### YOUR CODE ENDS HERE ####################################
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):

        #################### YOUR CODE BEGINS HERE ##################################

            hiddens, states = None                                # Get the hidden vector as well as the internal lstm state. :: hiddens: (batch_size, 1, hidden_size)
            hiddens = hiddens.squeeze(1)
            outputs = None                                       # Apply the linear layer just as in the forward function above :: outputs:  (batch_size, vocab_size)
            predicted = None                                     # Get the id of the predicted label. The predicted label is the label with maximum logits ::  predicted: (batch_size)

        #################### YOUR CODE ENDS HERE ####################################

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # Construct the input for the next timestep :: inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # Unsqueeze the first dimension which represents timesteps. :: inputs: (batch_size, 1, embed_size)

        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
