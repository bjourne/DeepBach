"""
@author: Gaetan Hadjeres
"""

from DatasetManager.metadata import FermataMetadata
from DatasetManager.helpers import END_SYMBOL, START_SYMBOL
from DeepBach.helpers import cuda_variable, to_numpy
from DeepBach.voice_model import VoiceModel
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import torch


# What does these constants do?
SEQUENCES_SIZE = 8

# Number of sixteenth notes per beat
SUBDIVISION = 4

# Number of parallel gibbs updates
BATCH_SIZE = 8

# timesteps_ticks is the number of ticks on which we unroll
# the LSTMs it is also the padding size
TIMESTEP_TICKS = SEQUENCES_SIZE * SUBDIVISION // 2

def random_chorale(length, n_tokens_per_voice):
    tensor = np.array(
        [np.random.randint(n_tokens, size = length)
         for n_tokens in n_tokens_per_voice])
    return torch.from_numpy(tensor).long().clone()

class DeepBach:
    def __init__(self,
                 dataset,
                 note_embedding_dim,
                 meta_embedding_dim,
                 num_layers,
                 lstm_hidden_size,
                 dropout_lstm,
                 linear_hidden_size):
        self.dataset = dataset
        self.num_voices = self.dataset.num_voices
        self.num_metas = len(self.dataset.metadatas) + 1
        self.activate_cuda = torch.cuda.is_available()

        self.voice_models = [VoiceModel(
            self.dataset,
            main_voice_index,
            note_embedding_dim,
            meta_embedding_dim,
            num_layers,
            lstm_hidden_size,
            dropout_lstm,
            linear_hidden_size)
            for main_voice_index in range(4)]

    def cuda(self, main_voice_index=None):
        if self.activate_cuda:
            if main_voice_index is None:
                for voice_index in range(self.num_voices):
                    self.cuda(voice_index)
            else:
                self.voice_models[main_voice_index].cuda()

    # Utils
    def load(self, main_voice_index=None):
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.load(main_voice_index=voice_index)
        else:
            self.voice_models[main_voice_index].load()

    def save(self, main_voice_index=None):
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.save(main_voice_index=voice_index)
        else:
            self.voice_models[main_voice_index].save()

    def train(self, main_voice_index=None,
              **kwargs):
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.train(main_voice_index=voice_index, **kwargs)
        else:
            voice_model = self.voice_models[main_voice_index]
            if self.activate_cuda:
                voice_model.cuda()
            optimizer = optim.Adam(voice_model.parameters())
            voice_model.train_model(optimizer=optimizer, **kwargs)

    def eval_phase(self):
        for voice_model in self.voice_models:
            voice_model.eval()

    def train_phase(self):
        for voice_model in self.voice_models:
            voice_model.train()

    def generation(self, temperature=1.0,
                   num_iterations=None,
                   length=160,
                   fermatas=None):
        """

        :param temperature:
        :param batch_size_per_voice:
        :param num_iterations:
        :param tensor_chorale:
        :param tensor_metadata:
        :param fermatas: list[Fermata]
        the portion of the score on which we apply the pseudo-Gibbs algorithm
        :return: tuple (
            generated_score,
            tensor_chorale (num_voices, chorale_length),
            tensor_metadata (num_voices, chorale_length, num_metadata)
        )
        """
        self.eval_phase()

        n_tokens_per_voice = [len(n2i)
                              for n2i in self.dataset.note2index_dicts]

        tensor_chorale = random_chorale(length, n_tokens_per_voice)

        # initialize metadata
        it = self.dataset.corpus_it_gen().__iter__()
        test_chorale = next(it)
        test_chorale = next(it)
        test_chorale = next(it)
        test_chorale = next(it)
        test_chorale = next(it)

        md = []
        for metadata in self.dataset.metadatas:
            a = metadata.evaluate(test_chorale, SUBDIVISION)
            seq_metadata = torch.from_numpy(a).long().clone()
            square_metadata = seq_metadata.repeat(4, 1)
            md.append(square_metadata[:, :, None])
        chorale_length = int(test_chorale.duration.quarterLength \
                             * SUBDIVISION)
        voice_id_metada = torch.from_numpy(np.arange(4)).long().clone()
        square_metadata = torch.transpose(
            voice_id_metada.repeat(chorale_length, 1), 0, 1)
        md.append(square_metadata[:, :, None])
        tensor_metadata = torch.cat(md, 2)

        # todo do not work if metadata_length_ticks > sequence_length_ticks
        tensor_metadata = tensor_metadata[:, :length, :]

        if fermatas is not None:
            tensor_metadata = self.dataset.set_fermatas(tensor_metadata,
                                                        fermatas)

        # Pad left and right
        left = np.array([n2i[START_SYMBOL]
                         for n2i in self.dataset.note2index_dicts])
        left = torch.from_numpy(left).long().clone()
        left = left.repeat(TIMESTEP_TICKS, 1).transpose(0, 1)

        right = np.array([n2i[END_SYMBOL]
                          for n2i in self.dataset.note2index_dicts])
        right = torch.from_numpy(right).long().clone()
        right = right.repeat(TIMESTEP_TICKS, 1).transpose(0, 1)
        tensor_chorale = torch.cat([left, tensor_chorale, right], 1)

        n_metadata = tensor_metadata.shape[2]

        left = np.zeros((4, TIMESTEP_TICKS, n_metadata))
        left = torch.from_numpy(left).long().clone()

        right = np.zeros((4, TIMESTEP_TICKS, n_metadata))
        right = torch.from_numpy(right).long().clone()

        tensor_metadata_padded = torch.cat([left, tensor_metadata, right],
                                           1)

        tensor_chorale = self.parallel_gibbs(
            tensor_chorale,
            tensor_metadata_padded,
            num_iterations=num_iterations,
            temperature=temperature)

        # get fermata tensor
        for metadata_index, metadata in enumerate(self.dataset.metadatas):
            if isinstance(metadata, FermataMetadata):
                break

        print('solved tensor', tensor_chorale)
        score = self.dataset.tensor_to_score(
            tensor_score=tensor_chorale,
            fermata_tensor=tensor_metadata[:, :, metadata_index])

        return score, tensor_chorale, tensor_metadata

    def parallel_gibbs(self, chorale, metadata,
                       num_iterations=1000, temperature=1.):
        """
        Parallel pseudo-Gibbs sampling
        chorale and metadata are padded with
        timesteps_ticks START_SYMBOLS before,
        timesteps_ticks END_SYMBOLS after

        :param chorale: (num_voices, chorale_length) tensor
        :param metadata: (num_voices, chorale_length) tensor
        :param num_iterations: number of Gibbs sampling iterations
        :param temperature: final temperature after simulated annealing
        :return: (num_voices, chorale_length) tensor
        """

        time_index_range_ticks = [TIMESTEP_TICKS,
                                  chorale.shape[1] - TIMESTEP_TICKS]

        # add batch_dimension
        chorale = chorale.unsqueeze(0)
        chorale_no_cuda = chorale.clone()
        metadata = metadata.unsqueeze(0)

        # to variable
        chorale = cuda_variable(chorale)
        metadata = cuda_variable(metadata)

        min_temperature = temperature
        temperature = 1.1

        # Main loop
        for iteration in tqdm(range(num_iterations)):
            # annealing
            temperature = max(min_temperature, temperature * 0.9993)
            # print(temperature)
            ticks_per_row = np.zeros((4, BATCH_SIZE), dtype = np.int)
            probas = {}

            for row in range(4):
                batch_notes = []
                batch_metas = []

                #ticks_per_row[row] = []
                voice_model = self.voice_models[row]

                # create batches of inputs
                for col in range(BATCH_SIZE):
                    tick = np.random.randint(*time_index_range_ticks)
                    ticks_per_row[row, col] = tick
                    c1 = tick - TIMESTEP_TICKS
                    c2 = tick + TIMESTEP_TICKS
                    chorale_slice = chorale[:, :, c1:c2]
                    notes, _ = voice_model.preprocess_notes(
                        chorale_slice,
                        TIMESTEP_TICKS)
                    metadata_slice = metadata[:, :, c1:c2, :]
                    metas = voice_model.preprocess_metas(metadata_slice,
                                                         TIMESTEP_TICKS)

                    batch_notes.append(notes)
                    batch_metas.append(metas)

                # reshape batches
                batch_notes = list(map(list, zip(*batch_notes)))
                batch_notes = [torch.cat(lcr) if lcr[0] is not None
                               else None
                               for lcr in batch_notes]
                batch_metas = list(map(list, zip(*batch_metas)))
                batch_metas = [torch.cat(lcr) for lcr in batch_metas]

                # make all estimations
                probas[row] = voice_model.forward(batch_notes,
                                                  batch_metas)
                probas[row] = nn.Softmax(dim=1)(probas[row])

            # update all predictions
            for row in range(4):
                for batch_index in range(BATCH_SIZE):
                    probas_pitch = probas[row][batch_index]

                    probas_pitch = to_numpy(probas_pitch)

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(
                        np.exp(probas_pitch)) - 1e-7

                    # avoid non-probabilities
                    probas_pitch[probas_pitch < 0] = 0

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))
                    col = ticks_per_row[row, batch_index]
                    chorale_no_cuda[0, row, col] = int(pitch)

            chorale = cuda_variable(chorale_no_cuda.clone())
            if iteration % 50 == 0:
                print(chorale_no_cuda[0, :, TIMESTEP_TICKS:-TIMESTEP_TICKS])

        return chorale_no_cuda[0, :, TIMESTEP_TICKS:-TIMESTEP_TICKS]
