import argparse
from ui import build_interface

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DeepCount App")
    parser.add_argument("--model", type=str, required=True, help="Path to the model (.pth file)")
    args = parser.parse_args()

    # Build and launch the app
    demo = build_interface(model_path=args.model)
    demo.launch(show_error=True, inline=False, inbrowser=True)

if __name__ == "__main__":
    main()
