import gymnasium as gym
import numpy as np

from sf_examples.nethack.utils.nle_tokenizer.tokenizer import (
    NLE_TOKENIZER,
    NLE_TOKENIZER_TUPLE_2_TOK,
    NLE_TOKENIZER_TOK_2_STR, DND_TOKENIZER, dnd_tokenizer_process_word, DND_DETOKENIZER, DND_TUPLE_TOKENIZER,
    DND_PUNCTUATION,
)

class NLETokenizerWrapper(gym.Wrapper):
    def __init__(
            self,
            env,
            tokenizer_name,
            max_token_length = 16,
            max_inv_items = 24,
            max_menu_items = 10,
            max_spells = 3,
            max_overview_rows = 15,
    ):
        super().__init__(env)
        self.tokenizer_name = tokenizer_name

        if tokenizer_name == "nle":
            self.tokenizer = NLE_TOKENIZER
            self.detokenizer = NLE_TOKENIZER_TOK_2_STR
            self.tuple_tokenizer = NLE_TOKENIZER_TUPLE_2_TOK
        elif tokenizer_name == "dnd":
            self.tokenizer = DND_TOKENIZER
            self.detokenizer = DND_DETOKENIZER
            self.tuple_tokenizer = DND_TUPLE_TOKENIZER
        else:
            raise ValueError(f"Unknown tokenizer {tokenizer_name}")

        self.max_token_length = max_token_length
        self.max_inv_items = max_inv_items
        self.max_menu_items = max_menu_items
        self.max_spells = max_spells
        self.max_overview_rows = max_overview_rows

        max_token = np.max(list(self.tokenizer.values()))

        obs_spaces = {
            'msg_tok': gym.spaces.Box(0, max_token, shape=(self.max_token_length,), dtype=np.uint16),
        }

        obs_spaces.update(
            [(k, self.env.observation_space[k]) for k in self.env.observation_space]
        )

        if 'inv_strs' in obs_spaces.keys():
            obs_spaces.pop('inv_strs')
            obs_spaces["inv_tok"] = gym.spaces.Box(0, max_token, shape=(self.max_inv_items, self.max_token_length), dtype=np.uint16)

        if 'menu_strs' in obs_spaces.keys():
            obs_spaces.pop('menu_strs')
            obs_spaces['menu_tok'] = gym.spaces.Box(0,max_token, shape=(self.max_menu_items, self.max_token_length), dtype=np.uint16)

        if 'spells_strs' in obs_spaces.keys():
            obs_spaces.pop('spells_strs')
            obs_spaces['spells_tok'] = gym.spaces.Box(0,max_token, shape=(self.max_spells, self.max_token_length), dtype=np.uint16)

        if 'overview_strs' in obs_spaces.keys():
            obs_spaces.pop('overview_strs')
            obs_spaces['overview_tok'] = gym.spaces.Box(0,max_token, shape=(self.max_overview_rows, self.max_token_length), dtype=np.uint16)


        self.inv_tok_buffer = np.zeros((self.max_inv_items, self.max_token_length))
        self.menu_tok_buffer = np.zeros((self.max_menu_items, self.max_token_length))
        self.message_tok_buffer = np.zeros(self.max_token_length)
        self.spells_tok_buffer = np.zeros((self.max_spells, self.max_token_length))
        self.overview_tok_buffer = np.zeros((self.max_overview_rows, self.max_token_length))

        self.inv_checksum = -1
        self.menu_checksum = -1
        self.message_checksum = -1
        self.spells_checksum = -1
        self.overview_checksum = -1

        self.observation_space = gym.spaces.Dict(obs_spaces)


    def _tokenize_matrix(self, inputs, buf):
        # add a delimiter character at the end of each row
        delim = ord('\n')
        inputs[:, -1] = delim
        # filter out zeros, there are usually a lot
        if self.tokenizer_name == "dnd":
            for p in DND_PUNCTUATION:
                inputs = inputs * (inputs != p)
        inputs = inputs[inputs != 0]
        # this is now a 1-D array, split by the delimiters we added above
        inputs = np.split(inputs, np.where(inputs == delim)[0])
        inputs = [item[(1 if i > 0 else 0):] for i, item in enumerate(inputs) if len(item) > 1]
        inputs = [np.split(item, np.where(item == ord(' '))[0]) for item in inputs]

        for i, item in enumerate(inputs):
            tokenized_item = np.array([self.tuple_tokenizer[tuple(x)] for x in item])
            tokenized_item = tokenized_item[tokenized_item != 0]
            n_tokens = min(self.max_token_length, len(tokenized_item))
            buf[i][:n_tokens] = tokenized_item[:n_tokens]


    def _tokenize(self, obs, debug=False):

        # tokenize the message
        if 'msg_tok' in self.observation_space.keys():
            message = obs['message']

            message_checksum = message.sum()
            if message_checksum != self.message_checksum:
                self.message_checksum = message_checksum

                if self.tokenizer_name == "dnd":
                    for p in DND_PUNCTUATION:
                        message = message * (message != p)
                message = message[message != 0]
                message = np.split(message, np.where(message == ord(' '))[0])
                message_tok_ind = np.array([self.tuple_tokenizer[tuple(x)] for x in message])[:self.max_token_length]
                message_tok_ind = np.sort(message_tok_ind[message_tok_ind != 0])
                self.message_tok_buffer = np.pad(message_tok_ind, (0, self.max_token_length - len(message_tok_ind)))

            obs["msg_tok"] = self.message_tok_buffer

            if debug:
                self._print_msg(obs["message"], obs["msg_tok"])

        # tokenize spells
        if 'spells_tok' in self.observation_space.keys():
            spells_checksum = obs["spells_strs"].sum()

            if spells_checksum != self.spells_checksum:
                self.spells_checksum = spells_checksum
                self.spells_tok_buffer.fill(0)
                self._tokenize_matrix(obs['spells_strs'][:self.max_spells], self.spells_tok_buffer)

            obs['spells_tok'] = self.spells_tok_buffer

            if debug:
                print('### original spells: ###')
                self._print_inv_strs(obs['spells_strs'])
                print('### tokenized spells: ###')
                self._print_inv_tok(self.spells_tok_buffer)

        # tokenize dungeon overview
        if 'overview_tok' in self.observation_space.keys():
            overview_checksum = obs["overview_strs"].sum()

            if overview_checksum != self.overview_checksum:
                self.overview_checksum = overview_checksum
                self.overview_tok_buffer.fill(0)
                self._tokenize_matrix(obs['overview_strs'][:self.max_overview_rows], self.overview_tok_buffer)

            obs['overview_tok'] = self.overview_tok_buffer

            if debug:
                print('### original overview: ###')
                self._print_inv_strs(obs['overview_strs'])
                print('### tokenized overview: ###')
                self._print_inv_tok(self.overview_tok_buffer)


        # tokenize the inventory
        if 'inv_tok' in self.observation_space.keys():
            inv_checksum = obs["inv_strs"].sum()

            if inv_checksum != self.inv_checksum:
                self.inv_checksum = inv_checksum

                self.inv_tok_buffer.fill(0)
                self._tokenize_matrix(obs['inv_strs'][:self.max_inv_items], self.inv_tok_buffer)

            obs['inv_tok'] = self.inv_tok_buffer

            if debug:
                print('### original inventory: ###')
                self._print_inv_strs(obs['inv_strs'])
                print('### tokenized inventory: ###')
                self._print_inv_tok(self.inv_tok_buffer)


        # tokenize menu
        if 'menu_tok' in self.observation_space.keys():
            menu_checksum = obs["menu_strs"].sum()

            if menu_checksum != self.menu_checksum:
                self.menu_checksum = menu_checksum

                self.menu_tok_buffer.fill(0)
                if obs['menu_strs'].sum() > 0:
                    self._tokenize_matrix(obs['menu_strs'][:self.max_menu_items], self.menu_tok_buffer)

                    if debug:
                        print('### message ###')
                        print(bytes(obs['message']).decode('latin-1'))
                        print('### original menu items: ###')
                        self._print_inv_strs(obs['menu_strs'])
                        print('### tokenized menu items: ###')
                        self._print_inv_tok(self.menu_tok_buffer)

            obs['menu_tok'] = self.menu_tok_buffer

        return obs

    def _print_msg(self, msg_chrs, msg_tok):
        msg_str = "".join([chr(c) for c in msg_chrs if c != 0])
        print("Original message:", msg_str)
        s = ''
        for tok in msg_tok:
            s += self.detokenizer[int(tok)] + " "
        print("Tokenized message:", s)
        print()

    def _print_inv_tok(self, inv_tok):
        s = ''
        for item in inv_tok:
            if item.sum() != 0:
                for x in item:
                    s += self.detokenizer[int(x)] + ', '
                s += '\n'
        print(s)

    def _print_inv_strs(self, inv_strs):
        s = ''
        for item in inv_strs:
            if item[:-1].sum() != 0:
                print(bytes(item).decode('latin-1'))

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        obs = self._tokenize(obs)
        return obs, reward, term, trun, info


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._tokenize(obs)
        return obs, info

