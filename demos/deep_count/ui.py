# Graphical user interface
import gradio as gr
from constants import (
    TITLE, AUTHOR, DESCRIPTION, WARNING_MESSAGE, TECH_COPY_TEXT, APP_THEME_COLOR,
    EXAMPLES, MAX_ALPHA, MIN_ALPHA, DEFAULT_ALPHA, DEFAULT_STEP, MAX_DIM,
    CONF_MED, DEFAULT_BOX_COLOR, DEFAULT_DOWNSCALE, COLOR_CHOICES, CONF_CHOICES
)
from backend import detect_items, show_analyzing, prev_split, next_split, fresh_state, downscale_status

def build_interface():
    
    # Define theme
    theme = gr.themes.Ocean(
        primary_hue=APP_THEME_COLOR,
        secondary_hue=APP_THEME_COLOR)
    css = """
    /* Only hide number input inside a specific slider container */
    .gr-slider-container input[type=number] {
        display: none !important;
    }
    """

    # Gradio interface setup
    with gr.Blocks(theme=theme, css=css) as demo:
        
        # Title / author / description
        gr.Markdown(f"<h1>{TITLE}</h1>")
        gr.Markdown(AUTHOR)
        gr.Markdown(DESCRIPTION)

        # Warning message
        if not(DEFAULT_DOWNSCALE):
            gr.Markdown("<hr>")
            gr.Markdown(WARNING_MESSAGE)
            gr.Markdown("<hr>")

        # Initialize states
        state = gr.State(fresh_state())

        # Whole interface is a single row of two columns: input interface and output interface
        with gr.Row():

            # Input interface
            with gr.Column():

                # Controls
                with gr.Row(min_height="110px"):                
                    conf_input = gr.Dropdown(
                        label="Detection Sensitivity",
                        choices=CONF_CHOICES,
                        value=CONF_MED
                    )
                    color_input = gr.Dropdown(
                        label="Bounding Box Color",
                        choices=COLOR_CHOICES,
                        value=DEFAULT_BOX_COLOR
                    )
                    dimm_input = gr.Slider(
                        label="Dimm Background",
                        minimum=1-MAX_ALPHA, maximum=1-MIN_ALPHA,
                        step=DEFAULT_STEP, value=1-DEFAULT_ALPHA
                    )
                
                # Downscale Image/Clear
                with gr.Row():
                    #downscale_input = gr.Checkbox(
                    #    label=f"Downscale Image (Width: {MAX_DIM}px)",
                    #    value=DEFAULT_DOWNSCALE
                    #)
                    clear_btn = gr.ClearButton()
                
                # Load image
                image_input = gr.Image(
                    type="pil",
                    interactive=True,
                    label="Original Image",
                    sources=["upload", "clipboard"]
                )

                # Show example images
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[image_input],
                    label="Example Images"
                    )

            # Output interface
            with gr.Column():

                # Items detected output
                with gr.Row(min_height="110px"):
                    count_output = gr.Label(label="Number of Items", value="")
                    conf_output = gr.Label(label="Item Confidence", value="")
                
                # Split items
                with gr.Row():

                    # Previous and next buttons
                    prev_btn = gr.Button("Previous Item")
                    next_btn = gr.Button("Next Item")
                    
                    # Download button
                    #download_btn = gr.Button("Download Selection")
                    #download_output = gr.File(label="Download Processed Image", file_path=None)

                # Image with bounding boxes
                image_output = gr.Image(
                    type="pil",
                    label="Identified Items",
                    sources=[],
                    interactive=False
                )

        #Downscale checkbox
        #downscale_input.change(
        #    downscale_status,
        #    inputs=[downscale_input, state],
        #    outputs=[state]
        #).then(
        #    detect_items,
        #    inputs=[image_input, conf_input, color_input, dimm_input, state],
        #    outputs=[image_output, count_output, conf_output, state],
        #    show_progress=False
        #    )

        # Clear-button event
        clear_btn.click(
            lambda: (None, None, "", "", fresh_state()),
            inputs=[],
            outputs=[image_input, image_output, count_output, conf_output, state],
            show_progress=False
        )#.then(
        #    downscale_status,
        #    inputs=[downscale_input, state],
        #    outputs=[state]
        #)

        # Navigate bounding boxes
        prev_btn.click(
            prev_split,
            inputs=[state],
            outputs=[state]
        ).then(
            detect_items,
            inputs=[image_input, conf_input, color_input, dimm_input, state],
            outputs=[image_output, count_output, conf_output, state],
            show_progress=False
            )
        
        next_btn.click(
            next_split,
            inputs=[state],
            outputs=[state]
        ).then(
            detect_items,
            inputs=[image_input, conf_input, color_input, dimm_input, state],
            outputs=[image_output, count_output, conf_output, state],
            show_progress=False
            )
        
        # Event to generate downloadable image
        #download_btn.click(
        #    generate_item,
        #    inputs=[image_input, conf_input, color_input, dimm_input, state],
        #    outputs=download_output,
        #    show_progress=False
        #)

        # Events when changing parameters
        for input_widget in (conf_input, color_input, dimm_input):
            input_widget.change(
                detect_items,
                inputs=[image_input, conf_input, color_input, dimm_input, state],
                outputs=[image_output, count_output, conf_output, state],
                show_progress=False
                )

        # Display 'Analyzing...' during prediction
        image_input.change(
            show_analyzing,
            inputs=[image_input, state],
            outputs=[image_output, count_output, conf_output, state],
            show_progress=False
            )

        # Event when loading a new image
        image_input.change(
            detect_items,
            inputs=[image_input, conf_input, color_input, dimm_input, state],
            outputs=[image_output, count_output, conf_output, state],
            show_progress=False
            )

        # Copyright and technical information
        gr.Markdown("<hr>")
        gr.Markdown(TECH_COPY_TEXT)
        gr.Markdown("<hr>")

    return demo
