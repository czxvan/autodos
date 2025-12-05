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


def save_results(
    config: AutoDoSConfig,
    successful_prompts: list[str],
    attack: AutoDoSAttack,
) -> None:
    """Save attack results to disk.
    
    Args:
        config: AutoDoS configuration
        successful_prompts: List of successful attack prompts
        attack: AutoDoS attack instance with history
    """
    # Create output directory
    output_dir = Path(config.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save successful prompts
    if successful_prompts:
        prompts_file = output_dir / f"successful_prompts_{timestamp}.txt"
        with open(prompts_file, 'w', encoding='utf-8') as f:
            f.write(f"# AutoDoS Successful Prompts\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total: {len(successful_prompts)}\n\n")
            for i, prompt in enumerate(successful_prompts, 1):
                f.write(f"{'='*60}\n")
                f.write(f"PROMPT {i}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{prompt}\n\n")
        print(f"\n✓ Saved {len(successful_prompts)} successful prompts to: {prompts_file}")
    
    # Save attack history
    if config.output.save_intermediate and attack.attack_history:
        history_file = output_dir / f"attack_history_{timestamp}.json"
        history_data = [result.model_dump() for result in attack.attack_history]
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved attack history to: {history_file}")
    
    # Save summary
    summary_file = output_dir / f"summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"AutoDoS Attack Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Target: {config.target.function_description[:100]}...\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Model: {config.agents.target['model']}\n")
        f.write(f"  Backend: {config.agents.target['backend_type']}\n")
        f.write(f"  Subproblems: {config.attack.n_subproblems}\n")
        f.write(f"  Optimization Iterations: {config.attack.optimize_iterations}\n")
        f.write(f"  Optimization Streams: {config.attack.n_optimization_streams}\n")
        f.write(f"  Max Concurrent Requests: {config.attack.max_concurrent_requests}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Total Attempts: {len(attack.attack_history)}\n")
        f.write(f"  Successful Prompts: {len(successful_prompts)}\n")
        f.write(f"  Success Rate: {len(successful_prompts) / len(attack.attack_history) * 100:.1f}%\n" if attack.attack_history else "  Success Rate: N/A\n")
    
    print(f"✓ Saved summary to: {summary_file}\n")


async def run_attack(config_path: str) -> None:
    """Run AutoDoS attack.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = AutoDoSConfig.from_yaml(config_path)
    
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
    print(f"{'='*60}\n")
    
    # Run attack
    logger.info("Initializing AutoDoS attack...")
    attack = AutoDoSAttack(config)
    
    try:
        logger.info("Starting attack execution...")
        start_time = datetime.now()
        
        successful_prompts = await attack.arun()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        print(f"\n{'='*60}")
        print(f"Attack Complete")
        print(f"{'='*60}")
        print(f"Duration: {duration:.1f}s")
        print(f"Total Attempts: {len(attack.attack_history)}")
        print(f"Successful Prompts: {len(successful_prompts)}")
        
        if successful_prompts:
            print(f"\n✓ SUCCESS: Found {len(successful_prompts)} working prompts!")
            print(f"\nFirst successful prompt preview:")
            print(f"{'-'*60}")
            print(f"{successful_prompts[0][:200]}...")
            print(f"{'-'*60}")
        else:
            print(f"\n✗ No successful prompts found.")
            print(f"Consider adjusting parameters or trying a different model.")
        
        # Save results
        print(f"\nSaving results...")
        save_results(config, successful_prompts, attack)
        
        logger.info(f"Attack completed in {duration:.1f}s")
        
    except KeyboardInterrupt:
        logger.warning("\n\nAttack interrupted by user")
        print(f"\n{'='*60}")
        print(f"Attack Interrupted")
        print(f"{'='*60}")
        print(f"Partial results will be saved...\n")
        save_results(config, [], attack)
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
  # Run with default config
  python main.py --config configs/deepseek_attack.yaml
  
  # Run with custom config
  python main.py --config my_config.yaml
  
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
    
    # Run attack
    try:
        asyncio.run(run_attack(args.config))
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == '__main__':
    main()
