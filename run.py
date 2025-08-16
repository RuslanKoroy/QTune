import os
import sys
import subprocess

def main():
    print("🚀 Starting QTune...")
    print("Please wait while the application initializes...\n")
    
    # Run the main application
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running QTune Studio: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 QTune Studio stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()