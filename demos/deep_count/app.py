# Main function
from ui import build_interface

demo = build_interface()

if __name__ == "__main__":
    demo.launch(show_error=True, inline=False, inbrowser=True)