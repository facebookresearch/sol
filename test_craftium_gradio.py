"""
Craftium Demo - Gradio Application
"""

import craftium  # noqa: F401
import gradio as gr
import gymnasium as gym
from PIL import Image
from sf_examples.craftium.wrappers.extra_info_wrapper import ExtraInfoWrapper

'''
KEYMAP_CHOPTREE = {
    "z": 0,  # nop (also releasing any key)
    "w": 1,  # forward
    "space": 2,  # jump
    "d": 3,  # dig
    "Right": 4,  # mouse x-
    "Left": 5,  # mouse x+
    "Up": 6,  # mouse y+
    "Down": 7,  # mouse y-
}
'''

KEYMAP = {
    "w": 1,  # forward
    "space": 5,  # jump
    "d": 7,  # dig
    "Right": 14,  # mouse x-
    "Left": 15,  # mouse x+
    "Up": 16,  # mouse y+
    "Down": 17,  # mouse y-
}


KEYMAP_CHOPTREE = {
    "w": 1,  # forward
    "space": 5,  # jump
    "d": 7,  # dig
    "Right": 14,  # mouse x-
    "Left": 15,  # mouse x+
    "Up": 16,  # mouse y+
    "Down": 17,  # mouse y-
    "slot_1": 9,   #9
    "slot_2": 10,   #10
    "slot_3": 11,   #11
    "slot_4": 12,   #12
    "slot_5": 13,   #13
    
}

'''
a = [
    "forward",  #1
    "backward", #2
    "left",     #3
    "right",    #4
    "jump",     #5
    "sneak",    #6 
    "dig",      #7
    "place",    #8
    "slot_1",   #9
    "slot_2",   #10
    "slot_3",   #11
    "slot_4",   #12
    "slot_5",   #13
    "mouse x+", #14
    "mouse x-", #15
    "mouse y+", #16
    "mouse y-"  #17
]
'''

ENV_NAMES = {
    "chop_tree": "Craftium/ChopTree-v0",
    "open_world": "Craftium/OpenWorld-v0",
}


class EnvironmentTerminatedError(Exception):
    """Raised when the environment episode is terminated."""

    pass


class CraftiumApp:
    """Gradio application for Craftium demo."""

    def __init__(self, env_name="chop_tree", fps=30, height=512, width=512) -> None:
        """Initialize the Craftium app."""
        self.app = self._create_app()
        self.env_name = env_name
        self.fps = fps
        self.delay = int(1000 / fps)
        self.height = height
        self.width = width

        self.env = None
        self.running = True
        self.current_obs = None
        self.current_action = 0  # becomes 0 (nop) when key is released

        # For saving
        self.obses, self.actions, self.rewards = [], [], []

    def _on_left(self):
        """Handle left button press."""
        print("Left pressed")
        self.current_action = KEYMAP_CHOPTREE["Left"]
        try:
            self.step_env()
            return self.get_current_frame(), f"Reward: {self.rewards[-1]}"
        except EnvironmentTerminatedError:
            return None, "terminated"

    def _on_right(self):
        """Handle right button press."""
        print("Right pressed")
        self.current_action = KEYMAP_CHOPTREE["Right"]
        try:
            self.step_env()
            return self.get_current_frame(), f"Reward: {self.rewards[-1]}"
        except EnvironmentTerminatedError:
            return None, "terminated"

    def _on_up(self):
        """Handle up button press."""
        print("Up pressed")
        self.current_action = KEYMAP_CHOPTREE["Up"]
        try:
            self.step_env()
            return self.get_current_frame(), f"Reward: {self.rewards[-1]}"
        except EnvironmentTerminatedError:
            return None, "terminated"

    def _on_down(self):
        """Handle down button press."""
        print("Down pressed")
        self.current_action = KEYMAP_CHOPTREE["Down"]
        try:
            self.step_env()
            return self.get_current_frame(), f"Reward: {self.rewards[-1]}"
        except EnvironmentTerminatedError:
            return None, "terminated"

    def _on_dig(self):
        """Handle dig button press."""
        print("Dig pressed")
        self.current_action = KEYMAP_CHOPTREE["d"]
        try:
            self.step_env()
            return self.get_current_frame(), f"Reward: {self.rewards[-1]}"
        except EnvironmentTerminatedError:
            return None, "terminated"

    def _on_jump(self):
        """Handle jump button press."""
        print("Jump pressed")
        self.current_action = KEYMAP_CHOPTREE["space"]
        try:
            self.step_env()
            return self.get_current_frame(), f"Reward: {self.rewards[-1]}"
        except EnvironmentTerminatedError:
            return None, "terminated"

    def _on_forward(self):
        """Handle forward button press."""
        print("Forward pressed")
        self.current_action = KEYMAP_CHOPTREE["w"]
        try:
            self.step_env()
            return self.get_current_frame(), f"Reward: {self.rewards[-1]}"
        except EnvironmentTerminatedError:
            return None, "terminated"

    def reset_env(self):
        obs, info = self.env.reset()
        self.current_obs = obs['img']
        self.obses = [obs]

    def step_env(self):
        obs, reward, terminated, truncated, info = self.env.step(self.current_action)
        print(info)

        self.obses.append(obs)
        self.actions.append(self.current_action)
        self.rewards.append(reward)

        self.current_obs = obs['img']

        if terminated:
            raise EnvironmentTerminatedError()

    def get_current_frame(self):
        if self.current_obs is not None:
            # Numpy(RGB) -> PIL Image
            img = Image.fromarray(self.current_obs)
        return img

    def _on_start_stop(self, environment: str, current_label: str):
        """Handle start/stop button press."""
        if current_label == "Start":
            print(f"Starting with environment: {environment}")
            self.env = gym.make(
                ENV_NAMES[environment],
                fps_max=self.fps,
                obs_width=self.height,
                obs_height=self.width,
                frameskip=4,
                sync_mode=True,
                #enable_inventory_obs=True,
            )
            compass_reward_directions = [int(180 * (i / 4)) for i in range(4)]
            
            self.env = ExtraInfoWrapper(self.env, compass_reward_directions, 300)
            self.reset_env()
            current_frame = self.get_current_frame()
            # Enable all buttons and change to Stop
            return (
                gr.update(value="Stop"),
                gr.update(value=f"Environment '{environment}' started"),
                current_frame,
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )
        else:
            print("Stopping")
            # Disable all buttons and change to Start
            return (
                gr.update(value="Start"),
                gr.update(value="Stopped"),
                None,
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

    def _create_app(self) -> gr.Blocks:
        """Create and configure the Gradio application."""
        with gr.Blocks(title="Craftium Playground") as app:
            ## Setup the main app
            gr.Markdown("# Craftium Playground")
            gr.Markdown(
                "Simple playground to explore [Craftium](https://github.com/mikelma/craftium) environment for MineCraft. Currently this works in devserver only."
            )

            with gr.Row():
                environment_dropdown = gr.Dropdown(
                    label="Select environment",
                    choices=list(ENV_NAMES.keys()),
                    value=list(ENV_NAMES.keys())[0],
                )
                self.status_box = gr.Textbox(label="Status", interactive=False)
                start_btn = gr.Button("Start")

            self.viewport = gr.Image(label="Image", type="numpy", interactive=False)

            with gr.Row():
                with gr.Column():
                    pass
                up_btn = gr.Button("Up", interactive=False)
                with gr.Column():
                    pass

            with gr.Row():
                left_btn = gr.Button("Left", interactive=False)
                down_btn = gr.Button("Down", interactive=False)
                right_btn = gr.Button("Right", interactive=False)

            with gr.Row():
                dig_btn = gr.Button("Dig", interactive=False)
                jump_btn = gr.Button("Jump", interactive=False)
                forward_btn = gr.Button("Forward", interactive=False)

            left_btn.click(fn=self._on_left, outputs=[self.viewport, self.status_box])
            right_btn.click(fn=self._on_right, outputs=[self.viewport, self.status_box])
            up_btn.click(fn=self._on_up, outputs=[self.viewport, self.status_box])
            down_btn.click(fn=self._on_down, outputs=[self.viewport, self.status_box])
            dig_btn.click(fn=self._on_dig, outputs=[self.viewport, self.status_box])
            jump_btn.click(fn=self._on_jump, outputs=[self.viewport, self.status_box])
            forward_btn.click(
                fn=self._on_forward, outputs=[self.viewport, self.status_box]
            )
            start_btn.click(
                fn=self._on_start_stop,
                inputs=[environment_dropdown, start_btn],
                outputs=[
                    start_btn,
                    self.status_box,
                    self.viewport,
                    up_btn,
                    down_btn,
                    left_btn,
                    right_btn,
                    dig_btn,
                    jump_btn,
                    forward_btn,
                ],
            )

        return app

    def launch(self) -> None:
        """Launch the Gradio application."""
        self.app.launch()


def main() -> None:
    """Main entry point for the application."""
    app = CraftiumApp()
    app.launch()


if __name__ == "__main__":
    main()
