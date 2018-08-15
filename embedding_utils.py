import numpy as np
import gensim

class Embedding():
	def __init__(self, sentence_list=[], max_sequence_length=10, dim_size=100, mode='load'):
		self.max_sequence_length = max_sequence_length #maximum sequence length of RNN
		self.dim_size = dim_size #dimension of embedding
		self.sentence_list = sentence_list
		self.mode = mode


		if self.mode == 'load':
			#load model
			self.model = gensim.models.Word2Vec.load('./data/models/trained_word2vec.model')

		elif self.mode == 'train':
			if len(sentence_list) == 0:
				raise ValueError('Sentence list is needed for training')
			else:
				#if file doesn't exist, then train the model
				#todo: adjust hyperparameter before training
				#link to docs: https://radimrehurek.com/gensim/models/word2vec.html
				self.model = gensim.models.Word2Vec(self.sentence_list,
						min_count=0)

				self.model.train(self.sentence_list, 
						total_examples=len(self.sentence_list),
						epochs=20)

				self.model.save('./data/models/trained_word2vec.model')

	def get_embedding_model(self):
		return self.model

	def get_vocab_size(self):
		word_vectors = self.model.wv
		return len(word_vectors.vocab)

	def get_dim_size(self):
		return self.dim_size

	def get_max_sequence(self):
		return self.max_sequence_length

	def get_embedding_matrix(self):
		word_vectors = self.model.wv
		#store each word's vector into a numpy array
		embedding_matrix = np.zeros((len(word_vectors.vocab), self.dim_size))
		for word, _ in word_vectors.vocab.items():
			#get the index of word
			index_of_word = word_vectors.vocab[word].index
			embedding_matrix[index_of_word] = word_vectors[word]

		return embedding_matrix