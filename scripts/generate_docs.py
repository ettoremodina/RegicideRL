import subprocess
import sys

def main():
    print("Generating documentation with pdoc...")
    
    # We document the game and agents modules
    modules_to_document = ["game", "agents", "ml_logger"]
    
    # Output directory
    output_dir = "docs"
    
    # Command to run
    cmd = [
        sys.executable, "-m", "pdoc",
        *modules_to_document,
        "-o", output_dir,
        "--docformat", "google"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Documentation generated successfully in ./{output_dir}/")
    except subprocess.CalledProcessError as e:
        print(f"Error generating documentation: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
