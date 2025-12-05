#!/usr/bin/env python3
"""AutoDoS main entry point."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

from autodos.config import AutoDoSConfig
from autodos.attack import AutoDoSAttack


def setup_logging(config: AutoDoSConfig) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        config: AutoDoS configuration
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = Path(config.logging.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_dir / f"autodos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    return logging.getLogger('autodos')


def save_successful_prompts(successful_prompts: list[str], run_dir: Path) -> None:
    """Save successful prompts to a separate file.
    
    Args:
        successful_prompts: List of successful attack prompts
        run_dir: Directory to save results
    """
    if successful_prompts:
        prompts_file = run_dir / "successful_prompts.txt"
        with open(prompts_file, 'w', encoding='utf-8') as f:
            f.write(f"# AutoDoS Successful Prompts\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total: {len(successful_prompts)}\n\n")
            for i, prompt in enumerate(successful_prompts, 1):
                f.write(f"{'='*60}\n")
                f.write(f"PROMPT {i}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{prompt}\n\n")
        print(f"✓ Saved {len(successful_prompts)} successful prompts")


async def run_attack(config_path: str, general_prompt_path: str = None) -> None:
    """Run AutoDoS attack.
    
    Args:
        config_path: Path to configuration YAML file
        general_prompt_path: Optional path to general prompt file
    """
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = AutoDoSConfig.from_yaml(config_path)
    
    # Override with command-line argument if provided
    if general_prompt_path:
        config.target.general_prompt_file = general_prompt_path
    
    # Setup logging
    logger = setup_logging(config)
    
    # Display configuration
    print(f"\n{'='*60}")
    print(f"AutoDoS Attack")
    print(f"{'='*60}")
    print(f"Model: {config.agents.target['model']}")
    print(f"Backend: {config.agents.target['backend_type']}")
    if 'provider' in config.agents.target:
        print(f"Provider: {config.agents.target['provider']}")
    print(f"\nAttack Parameters:")
    print(f"  Subproblems: {config.attack.n_subproblems}")
    print(f"  Optimization Iterations: {config.attack.optimize_iterations}")
    print(f"  Optimization Streams: {config.attack.n_optimization_streams}")
    print(f"  Max Concurrent Requests: {config.attack.max_concurrent_requests}")
    print(f"  Question Length: {config.attack.question_length}")
    print(f"\nTarget: {config.target.function_description[:100]}...")
    if config.target.general_prompt_file:
        print(f"\n⚡ Using existing general prompt: {config.target.general_prompt_file}")
        print(f"   (Skipping tree generation, will run optimization)")
    print(f"{'='*60}\n")
    
    # Run attack
    logger.info("Initializing AutoDoS attack...")
    attack = AutoDoSAttack(config)
    
    try:
        logger.info("Starting attack execution...")
        
        result = await attack.arun()
        
        # Display results
        print(f"\n{'='*60}")
        print(f"Attack Complete")
        print(f"{'='*60}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Total Attempts: {result.total_attempts}")
        print(f"Successful Prompts: {len(result.successful_prompts)}")
        print(f"Success Rate: {result.success_rate:.1f}%")
        print(f"Token Usage: {result.total_tokens:,} ({result.total_prompt_tokens:,} prompt + {result.total_completion_tokens:,} completion)")
        print(f"Estimated Cost: ${result.estimated_cost:.4f}")
        
        if result.success:
            print(f"\n✓ SUCCESS: Found {len(result.successful_prompts)} working prompts!")
            print(f"\nBest prompt preview:")
            print(f"{'-'*60}")
            print(f"{result.best_prompt[:200]}...")
            print(f"{'-'*60}")
        else:
            print(f"\n✗ No successful prompts found.")
            if result.error:
                print(f"Error: {result.error}")
            print(f"Consider adjusting parameters or trying a different model.")
        
        # Save results
        print(f"\nSaving results...")
        attack.save_results()
        save_successful_prompts(result.successful_prompts, attack._run_dir)
        print(f"\n{'='*60}")
        print(f"All results saved to: {attack._run_dir}")
        print(f"{'='*60}\n")
        
        logger.info(f"Attack completed in {result.duration_seconds:.1f}s")
        
    except KeyboardInterrupt:
        logger.warning("\n\nAttack interrupted by user")
        print(f"\n{'='*60}")
        print(f"Attack Interrupted")
        print(f"{'='*60}")
        print(f"Partial results will be saved...\n")
        attack.save_results()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Attack failed: {e}", exc_info=True)
        print(f"\n{'='*60}")
        print(f"Attack Failed")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"Check logs for details: {config.logging.output_dir}")
        print(f"{'='*60}\n")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AutoDoS - Automated Denial of Service Attack Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (tree generation + optimization)
  python main.py --config configs/deepseek_attack.yaml
  
  # Use existing general prompt (skip tree generation, run optimization)
  python main.py -c configs/deepseek_attack.yaml -g outputs/deepseek/run_XXX/general_prompt.txt
  
  # Get version info
  python main.py --version
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/deepseek_attack.yaml',
        help='Path to configuration YAML file (default: configs/deepseek_attack.yaml)'
    )
    
    parser.add_argument(
        '--general-prompt', '-g',
        type=str,
        default=None,
        help='Path to general prompt file (skips tree generation only)'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='AutoDoS 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {args.config}")
        print(f"Please provide a valid config file path.")
        sys.exit(1)
    
    # Validate prompt file if provided
    if args.general_prompt:
        general_prompt_path = Path(args.general_prompt)
        if not general_prompt_path.exists():
            print(f"Error: General prompt file not found: {args.general_prompt}")
            sys.exit(1)
    
    # Run attack
    try:
        asyncio.run(run_attack(args.config, args.general_prompt))
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == '__main__':
    main()
