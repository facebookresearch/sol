import gymnasium as gym
import numpy as np
import re
from nle.env.base import ASCII_SPACE, ASCII_ESC

from nle_patched.nle.nethack.actions import MiscAction


# Menu interaction wrapper as described in the blog post: https://iclr-blogposts.github.io/2026/blog/2026/revisiting-the-nle/

class MenuSelectionWrapper(gym.Wrapper):
    def __init__(self, env, do_gui_menu_selection=False, max_inv_choices=10, max_inv_str_length=80):
        super().__init__(env)
        self.max_inv_choices = max_inv_choices
        self.max_inv_str_length = max_inv_str_length
        self.do_gui_menu_selection = do_gui_menu_selection

        # Set action space to be (original_action x inventory_selection)
        inv_action_space = gym.spaces.Discrete(self.max_inv_choices)
        self.action_space = gym.spaces.Tuple((self.action_space, inv_action_space,))

        # Set observation space to include the set of inventory items to choose from, if any
        obs_spaces = {"menu_strs": gym.spaces.Box(0, 255, shape=(max_inv_choices, max_inv_str_length), dtype=np.uint8)}
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)

        self.menu_exit_vector = np.array(list("exit menu".encode("latin-1")), dtype=np.uint8)
        self.menu_exit_vector = np.pad(self.menu_exit_vector, (0, max_inv_str_length - len(self.menu_exit_vector)))

        self.menu_strs_np = np.zeros((self.max_inv_choices, self.max_inv_str_length), dtype=np.uint8)

        self.letter_choices = []
        self.letter_choice_rows = []
        self.last_menu_sign_col = None
        self.in_regex_menu = False
        self.in_gui_menu = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.letter_choices = []
        self.letter_choice_rows = []
        self.last_menu_sign_col = None
        self.in_regex_menu = False
        self.in_gui_menu = False
        self.menu_strs_np.fill(0)
        obs['menu_strs'] = self.menu_strs_np
        return obs, info

    def _process_gui_menu(self, tty_chars, debug=False):
        is_end = np.ones_like(tty_chars, dtype=bool)
        for i, c in enumerate("(end)"):
            if i == 0:
                tty_shifted = tty_chars
            else:
                tty_shifted = np.zeros_like(tty_chars)
                tty_shifted[:, :-i] = tty_chars[:, i:]

            is_end &= tty_shifted == ord(c)

            if not is_end.any():
                break

        if is_end.astype(float).sum() > 0:
            end_index_flat = np.argmax(is_end)
            end_anchor = (
                end_index_flat // 80,
                end_index_flat % 80,
            )
        else:
            end_anchor = None

        # If a menu is open, process it to extract the actionable rows (i.e. ignore headers)
        if end_anchor is not None:
            menu_tty = tty_chars[:end_anchor[0], end_anchor[1]:]

            row_index = 0
            for row_num, row in enumerate(menu_tty):
                row_str = "".join([chr(c) for c in row])

                if row_str[1:4] == " - " or row_str[1:4] == " + ":
                    row_length = min(self.max_inv_str_length, len(row))
                    self.menu_strs_np[row_index, :row_length] = row[:row_length]
                    self.letter_choices.append(row_str[0])
                    self.letter_choice_rows.append(row_num)
                    self.last_menu_sign_col = end_anchor[1] + 2  # The +/- is the 3rd character
                    row_index += 1

                if row_index >= self.max_inv_choices - 1:
                    break

            if len(self.letter_choices) > 0:
                self.menu_strs_np[row_index] = self.menu_exit_vector

            if debug:
                print("rows_np")
                for row in self.menu_strs_np:
                    print("".join([chr(c) for c in row]))

                print("")
                print("letter choices")
                print(self.letter_choices)
                print("")

        return len(self.letter_choices) > 0


    def _process_regex_menu(self, obs):

        # now check if the current message asks to select an object from inventory.
        # if so, we augment the observation with a list of these inventory items
        msg = bytes(obs['message']).decode('latin-1')

        letters = re.findall(r'\[([a-zA-Z-]+)', msg)

        # if we only have dash, this might be part of Elbereth macro, so skip
        # TODO: buggy
        if len(letters) > 0:
            if letters[0] == '-':
                letters = False

        if letters and "[yn" not in msg and "[rl]" not in msg:
            letters = letters[0]
            if '-' in letters[1:-1]:
                # when there are lots of choices, the message can be of the form: a-d to represent choices a,b,c,d.
                # we handle this case here.
                dash_idx = letters.index('-')
                start_char = letters[dash_idx - 1]
                end_char = letters[dash_idx + 1]
                letters = letters.replace('-', ''.join([chr(i) for i in range(ord(start_char) + 1, ord(end_char))]))
            letters = letters[:self.max_inv_choices - 1]
            inv_letters = bytes(obs['inv_letters']).decode('latin-1')
            menu_items = []
            for l in letters:
                if l in inv_letters:
                    menu_items.append(obs['inv_strs'][inv_letters.index(l)])
            if len(menu_items) > 0:
                menu_items.append(self.menu_exit_vector)
                self.menu_strs_np[:len(menu_items)] = np.stack(menu_items)
            self.letter_choices = letters

            return True
        else:
            self.letter_choices = []
            return False

    def _process_gui_action(self, action, debug=False):
        menu_row_selection = None
        menu_action = action[1]

        if debug:
            print("menu_action", menu_action)

        if menu_action < len(self.letter_choices):
            action_key = ord(self.letter_choices[menu_action])
            if action_key in self.env.unwrapped.actions:
                action = self.env.unwrapped.actions.index(action_key)
                menu_row_selection = self.letter_choice_rows[menu_action]
                if debug:
                    print("Selected action", action)
                    print("Menu row selection", menu_row_selection)
            else:
                action = self.env.unwrapped.actions.index(ASCII_ESC)
        else:
            action = self.env.unwrapped.actions.index(ASCII_ESC)

        return action, menu_row_selection

    def _process_regex_action(self, action):
        # decode the last message
        last_msg = bytes(self.env.unwrapped.last_observation[self.env.unwrapped._message_index]).decode('latin-1')

        # TODO: I don't think hyphen choices are correctly handled.
        # these refer to choices with the agent's body, such as engraving with finger
        # or wielding fists as a weapon.

        # check if we are selecting an object
        # "[yn" accounts for both "[yn]" and "[ynq]"
        # "[rl]" is for putting on rings
        if re.findall(r'\[([a-zA-Z]+)', last_msg) and "[yn" not in last_msg and "[rl]" not in last_msg:
            if not isinstance(action, int):
                action = action[1]
            # If we are running in play.py then this will be an int - allow us to interact with the inv selection
            # wrapper by selection 0-9
            else:
                action = action - 110
            if action < len(self.letter_choices):
                action_key = ord(self.letter_choices[action])
                if action_key in self.env.unwrapped.actions:
                    action = self.env.unwrapped.actions.index(action_key)
                else:
                    action = self.env.unwrapped.actions.index(ASCII_ESC)
            else:
                action = self.env.unwrapped.actions.index(ASCII_ESC)
        elif re.findall(r'\[\*\]', last_msg):
            # exit out if there are no valid choices
            action = self.env.unwrapped.actions.index(ASCII_ESC)
        else:
            # if not in selection menu, pick the first action tuple
            # it may be an int if we are playing in the terminal, so check for it here.
            if not isinstance(action, int):
                action = action[0]

        return action

    def step(self, action, debug=False):

        menu_row_selection = None
        if self.in_gui_menu:
            action, menu_row_selection = self._process_gui_action(action)
        elif self.in_regex_menu:
            action = self._process_regex_action(action)
        else:
            # if not in selection menu, pick the first action tuple
            # it may be an int if we are playing in the terminal, so check for it here.
            if not isinstance(action, int):
                action = action[0]

        obs, reward, term, trun, info = self.env.step(action)

        # If we are in a GUI multi-binary menu with "-" and "+", then we want to insert an ENTER
        if menu_row_selection is not None and not term and not trun:
            sign_char = obs["tty_chars"][menu_row_selection, self.last_menu_sign_col]
            if sign_char == ord('+'):
                obs, reward2, term, trun, info = self.env.step(self.env.unwrapped.actions.index(MiscAction.MORE))
                reward = reward + reward2

                if debug:
                    print("closing out of menu with automatic ENTER")
            else:
                if debug:
                    print("NOT auto-closing menu")

        self.menu_strs_np.fill(0)
        self.letter_choices = []
        self.letter_choice_rows = []
        self.last_menu_sign_col = None
        self.in_gui_menu = False
        self.in_regex_menu = False

        # First, we check if a GUI menu is open.
        # In the case of both GUI menu and regex menu, we give priority to the GUI menu
        if self.do_gui_menu_selection:
            self.in_gui_menu = self._process_gui_menu(obs["tty_chars"], debug=debug)

        if not self.in_gui_menu:
            self.in_regex_menu = self._process_regex_menu(obs)

        obs['menu_strs'] = self.menu_strs_np
        return obs, reward, term, trun, info






