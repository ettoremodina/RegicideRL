import subprocess
import sys
from ml_logger import get_logger, start_run

logger = get_logger(__name__)

def main():
    context = start_run("documentation")
    logger.info("Generating documentation with pdoc")
    
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
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip():
            logger.info("pdoc: %s", completed.stdout.strip())
        if completed.stderr.strip():
            logger.warning("pdoc: %s", completed.stderr.strip())
        result = {"status": "completed", "output_dir": output_dir}
        context.save_result("documentation.json", result)
        context.complete(result)
        logger.info("Documentation generated in %s", output_dir)
    except subprocess.CalledProcessError as e:
        context.fail(e)
        logger.exception("Documentation generation failed")
        raise
        
if __name__ == "__main__":
    main()
