#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
from src.utils import setup_environment, load_config

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Agents API Project CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup the project")
    
    # Example command
    example_parser = subparsers.add_parser("example", help="Run an example")
    example_parser.add_argument("name", help="Name of the example to run")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.add_argument("--path", help="Path to custom config file")
    
    # Multi-agent command
    multi_agent_parser = subparsers.add_parser("multi-agent", help="Run multi-agent examples")
    multi_agent_subparsers = multi_agent_parser.add_subparsers(dest="example_type", help="Type of multi-agent example to run")
    
    # Add simple multi-agent example
    simple_parser = multi_agent_subparsers.add_parser("simple", help="Run simple multi-agent example")
    
    # Add advanced multi-agent example
    advanced_parser = multi_agent_subparsers.add_parser("advanced", help="Run advanced multi-agent example")
    
    # Add verification multi-agent examples
    verification_parser = multi_agent_subparsers.add_parser("verification", help="Run verification multi-agent example")
    
    # Add SDK-style verification example
    sdk_verify_parser = multi_agent_subparsers.add_parser("sdk-verify", help="Run SDK-style verification example")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "setup":
        run_setup()
    elif args.command == "example":
        run_example(args.name)
    elif args.command == "config":
        show_config(args.path)
    elif args.command == "multi-agent":
        if args.example_type == "simple":
            run_example("multi_agent_example")
        elif args.example_type == "advanced":
            run_example("advanced_multi_agent")
        elif args.example_type == "verification":
            run_example("verification_example")
        elif args.example_type == "sdk-verify":
            run_example("verification_sdk_example")
        else:
            multi_agent_parser.print_help()
    else:
        parser.print_help()
        return 1
    
    return 0

def run_setup():
    """Setup the project"""
    print("Setting up the project...")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        # Create .env file from .env.example
        if os.path.exists(".env.example"):
            print("Creating .env file from .env.example...")
            with open(".env.example", "r") as example_file:
                content = example_file.read()
            
            with open(".env", "w") as env_file:
                env_file.write(content)
            
            print(".env file created. Please edit it to add your API key.")
        else:
            print("No .env.example file found. Creating empty .env file...")
            with open(".env", "w") as env_file:
                env_file.write("OPENAI_API_KEY=\n")
            
            print(".env file created. Please add your OpenAI API key.")
    else:
        print(".env file already exists.")
    
    print("Setup complete. You can now run examples with 'python cli.py example <name>'.")

def run_example(name):
    """Run an example by name"""
    # Check for the existence of the example file
    example_path = os.path.join("examples", f"{name}.py")
    if not os.path.exists(example_path):
        print(f"Error: Example '{name}' not found at {example_path}")
        return
    
    print(f"Running example: {name}")
    
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Setup environment variables
    setup_environment()
    
    # Import and run the example - handle both async and sync main functions
    try:
        example_module = __import__(f"examples.{name}", fromlist=["main"])
        
        if hasattr(example_module, "main"):
            main_func = example_module.main
            
            # Check if the main function is async
            if asyncio.iscoroutinefunction(main_func):
                # Run the async main function
                asyncio.run(main_func())
            else:
                # Run the regular main function
                main_func()
        else:
            print(f"Error: Example '{name}' has no main() function")
    except Exception as e:
        print(f"Error running example: {e}")

def show_config(config_path=None):
    """Show the current configuration"""
    try:
        config = load_config(config_path)
        print("Current configuration:")
        print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"Error loading configuration: {e}")

if __name__ == "__main__":
    # Add import for show_config
    import json
    
    sys.exit(main())